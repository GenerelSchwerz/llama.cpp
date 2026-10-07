#pragma once

#include "common.cuh"

#include <algorithm>
#include <deque>

struct ggml_cuda_reuse_pool : ggml_cuda_pool {
    char * base;
    size_t capacity;
    size_t used = 0;

    ggml_cuda_reuse_pool(char * base, size_t capacity) : base(base), capacity(capacity) {}

    void * alloc(size_t size, size_t * actual_size) override {
        GGML_ASSERT(size <= SIZE_MAX - 255);
        size = GGML_PAD(size, 256);
        GGML_ASSERT(size <= capacity - used);
        void * ptr = base + used;
        used += size;
        *actual_size = size;
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        GGML_ASSERT(size <= used);
        used -= size;
        GGML_ASSERT(ptr == base + used);
    }
};

struct ggml_cuda_reuse_group {
    int node, prepare, last, next;
    bool after;
    size_t size, offset = 0;
};

struct ggml_cuda_reuse_plan {
    std::vector<ggml_cuda_reuse_group> groups;
    std::vector<int> nodes;
    std::vector<int> starts;
    std::unordered_map<const ggml_tensor *, int> indices;
    size_t size = 0;

    ggml_cuda_reuse_plan(const ggml_cgraph * graph, const ggml_cuda_stream_context & streams,
            const std::vector<int> & keys, const std::vector<size_t> & sizes,
            bool (*overwritten)(const ggml_tensor *, const ggml_tensor *), bool singletons = false) {
        if (keys.empty()) { return; }
        const int n = graph->n_nodes;
        std::unordered_map<const ggml_tensor *, std::vector<int>> consumers;
        for (int i = 0; i < n; ++i) {
            if (keys[i] >= 0) { consumers[graph->nodes[i]->src[1]].push_back(i); }
        }
        if (std::none_of(consumers.begin(), consumers.end(), [](const auto & entry) { return entry.second.size() > 1; })) { return; }
        nodes.assign(n, -1);
        starts.assign(n, -1);
        const bool concurrent = !streams.concurrent_events.empty();
        std::vector<int> forks(concurrent ? n : 0, -1), joins(concurrent ? n : 0, -1), lanes(concurrent ? n : 0, 0);
        if (!streams.concurrent_events.empty()) {
            for (int i = 0; i < n; ++i) { indices[graph->nodes[i]] = i; }
        }
        for (const auto & entry : streams.concurrent_events) {
            const auto first = indices.find(entry.first), last = indices.find(entry.second.join_node);
            if (first == indices.end() || last == indices.end() || first->second >= last->second) {
                nodes.clear(); starts.clear(); indices.clear(); return;
            }
            const int fork = first->second, join = last->second;
            for (const auto & mapping : entry.second.stream_mapping) {
                const auto it = indices.find(mapping.first);
                if (it == indices.end()) { nodes.clear(); starts.clear(); indices.clear(); return; }
                const int i = it->second;
                if (i > fork && i < join) {
                    forks[i] = fork;
                    joins[i] = join;
                    lanes[i] = mapping.second;
                }
            }
        }
        const auto ready = [&](const ggml_tensor * tensor, int fork) {
            while (tensor->view_src) { tensor = tensor->view_src; }
            const auto it = indices.find(tensor);
            return it == indices.end() ? tensor->op == GGML_OP_NONE : it->second <= fork;
        };
        std::vector<int> members;
        groups.reserve(consumers.size());
        for (int i = 0; i < n; ++i) {
            if (keys[i] < 0 || nodes[i] >= 0) { continue; }
            const ggml_tensor * node = graph->nodes[i];
            const ggml_tensor * input = node->src[1];
            const ggml_tensor * ids = node->op == GGML_OP_MUL_MAT_ID ? node->src[2] : nullptr;
            const auto changed = [&](int j) {
                return overwritten(graph->nodes[j], input) || (ids && overwritten(graph->nodes[j], ids));
            };
            int prepare = i;
            bool after = false;
            if (concurrent && forks[i] >= 0) {
                const bool early = ready(input, forks[i]) && (!ids || ready(ids, forks[i]));
                bool immutable = true;
                for (int j = forks[i] + 1; j < joins[i]; ++j) {
                    if (early || j >= i || lanes[j] != lanes[i]) { immutable &= !changed(j); }
                }
                if (!immutable) { continue; }
                if (early) { prepare = forks[i]; after = true; }
            }
            members.clear();
            size_t size = 0;
            int last = i;
            for (int j = prepare + (after ? 1 : 0); j <= consumers.at(input).back(); ++j) {
                if (changed(j)) { break; }
                const ggml_tensor * other = graph->nodes[j];
                if (j < i || nodes[j] >= 0 || keys[j] != keys[i] || other->src[1] != input) { continue; }
                if ((other->op == GGML_OP_MUL_MAT_ID ? other->src[2] : nullptr) != ids ||
                        (ids && other->src[0]->ne[2] != node->src[0]->ne[2])) { continue; }
                if (concurrent && forks[i] >= 0 && !after && j < joins[i] && lanes[j] != lanes[i]) { continue; }
                if (concurrent && forks[j] >= 0) {
                    bool immutable = true;
                    for (int k = forks[j] + 1; k < joins[j]; ++k) {
                        if (k >= prepare + (after ? 1 : 0) || lanes[k] != lanes[i]) { immutable &= !changed(k); }
                    }
                    if (!immutable) { continue; }
                }
                members.push_back(j);
                size = std::max(size, sizes[j]);
                last = std::max(last, concurrent && joins[j] >= 0 ? joins[j] : j);
            }
            if (members.empty() || (!singletons && members.size() < 2) || size > SIZE_MAX - 255) { continue; }
            const int group = int(groups.size());
            groups.push_back({i, prepare, last, starts[prepare], after, GGML_PAD(size, 256)});
            starts[prepare] = group;
            for (const int member : members) { nodes[member] = group; }
        }
        pack();
    }

    void pack() {
        size = 0;
        std::fill(starts.begin(), starts.end(), -1);
        for (size_t i = 0; i < groups.size(); ++i) {
            auto & group = groups[i];
            group.next = starts[group.prepare];
            starts[group.prepare] = int(i);
        }
        struct slot { size_t size; int last; };
        std::vector<slot> slots;
        std::vector<int> order;
        order.reserve(groups.size());
        for (size_t i = 0; i < groups.size(); ++i) { order.push_back(int(i)); }
        std::sort(order.begin(), order.end(), [&](int a, int b) { return groups[a].prepare < groups[b].prepare; });
        for (const int i : order) {
            auto & group = groups[i];
            int best = -1;
            for (size_t j = 0; j < slots.size(); ++j) {
                if (slots[j].last < group.prepare && (best < 0 ||
                        std::max(slots[j].size, group.size) < std::max(slots[best].size, group.size))) { best = int(j); }
            }
            if (best < 0) {
                best = int(slots.size());
                slots.push_back({group.size, group.last});
            } else {
                slots[best].size = std::max(slots[best].size, group.size);
                slots[best].last = group.last;
            }
            group.offset = size_t(best);
        }
        std::vector<size_t> offsets;
        offsets.reserve(slots.size());
        for (const auto & slot : slots) {
            offsets.push_back(size);
            if (slot.size > SIZE_MAX - size) { groups.clear(); nodes.clear(); starts.clear(); size = 0; return; }
            size += slot.size;
        }
        for (auto & group : groups) { group.offset = offsets[group.offset]; }
    }
};

template <typename T>
struct ggml_cuda_reuse_inputs {
    struct item {
        ggml_cuda_reuse_pool pool;
        T input;

        item(char * base, size_t size) : pool(base, size), input(pool) {}
    };

    ggml_cuda_pool_alloc<char> storage;
    std::deque<item> groups;

    ggml_cuda_reuse_inputs(ggml_cuda_pool & pool, const ggml_cuda_reuse_plan & plan) : storage(pool) {
        if (plan.size) { storage.alloc(plan.size); }
        for (const auto & group : plan.groups) { groups.emplace_back(storage.get() + group.offset, group.size); }
    }

    T & operator[](int group) { return groups[group].input; }
};
