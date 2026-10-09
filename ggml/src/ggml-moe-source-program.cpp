#include "ggml-moe-source-program.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iterator>
#include <type_traits>
#include <unordered_map>

bool ggml_moe_source_prefill_partition(const std::vector<uint32_t> & counts, std::vector<int32_t> & classes, uint32_t cpu_row_budget) try {
    if (counts.size() != classes.size() || counts.size() > UINT32_MAX) { return false; }
    std::vector<uint32_t> candidates;
    for (uint32_t i = 0; i < counts.size(); ++i) {
        if (!counts[i] || classes[i] < 0 || classes[i] > 1) { return false; }
        if (classes[i] == 1 && counts[i] <= cpu_row_budget) { candidates.push_back(i); }
    }
    std::sort(candidates.begin(), candidates.end(), [&](uint32_t a, uint32_t b) {
        return counts[a] != counts[b] ? counts[a] < counts[b] : a < b;
    });
    // Admit small cohorts within the worker budget; stream the rest on the GPU.
    for (const auto i : candidates) {
        if (counts[i] > cpu_row_budget) { break; }
        cpu_row_budget -= counts[i];
        classes[i] = 2;
    }
    return true;
} catch (...) { return false; }

static bool source_statistics_sum(const uint64_t * counts, uint32_t n_experts, uint64_t observations) {
    if (!counts || !n_experts || n_experts > (1u << 22)) { return false; }
    uint64_t sum = 0;
    for (uint32_t i = 0; i < n_experts; ++i) {
        if (counts[i] > UINT64_MAX - sum) { return false; }
        sum += counts[i];
    }
    return sum == observations;
}

bool ggml_moe_source_statistics_valid(const ggml_backend_moe_source_statistics_v1 * statistics, uint32_t count) try {
    if (count > GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS * GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS || (count && !statistics)) { return false; }
    std::unordered_map<const ggml_tensor *, std::vector<uint32_t>> identities;
    uint64_t total = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const auto & source = statistics[i];
        if (!source.tensor || source.tensor->ne[2] <= 0 || source.tensor->ne[2] > INT32_MAX ||
                source.n_experts != uint64_t(source.tensor->ne[2]) || source.n_experts > (1u << 22) - total ||
                !source_statistics_sum(source.counts, source.n_experts, source.observations)) { return false; }
        total += source.n_experts;
        auto & domains = identities[source.tensor];
        if (std::find(domains.begin(), domains.end(), source.domain) != domains.end()) { return false; }
        domains.push_back(source.domain);
    }
    return true;
} catch (...) { return false; }

static bool source_scores_valid(const double * scores, uint32_t count) {
    if (!scores) { return true; }
    double sum = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (!std::isfinite(scores[i]) || scores[i] < 0) { return false; }
        sum += scores[i];
        if (!std::isfinite(sum)) { return false; }
    }
    return true;
}

bool ggml_moe_source_scores_valid(const ggml_backend_moe_source_statistics_v1 * statistics, const double * const * scores, uint32_t count) {
    if (!ggml_moe_source_statistics_valid(statistics, count) || (scores && !count)) { return false; }
    for (uint32_t i = 0; scores && i < count; ++i) {
        if (!scores[i] || !source_scores_valid(scores[i], statistics[i].n_experts)) { return false; }
    }
    return true;
}

bool ggml_moe_source_score_statistics(const std::vector<ggml_moe_profile_bank_statistics> & banks, std::vector<long double> & output) try {
    if (banks.empty() || banks.size() > GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS) { return false; }
    const auto experts = banks.front().n_experts;
    uint64_t payload = 0;
    for (const auto & bank : banks) {
        if (bank.n_experts != experts || !bank.payload_bytes || bank.payload_bytes > UINT64_MAX - payload ||
                !source_statistics_sum(bank.counts, experts, bank.observations) || !source_scores_valid(bank.scores, experts)) { return false; }
        payload += bank.payload_bytes;
    }
    std::vector<long double> scores(experts, 0);
    for (const auto & bank : banks) {
        if (!bank.observations && !bank.scores) { continue; }
        const auto weight = static_cast<long double>(bank.payload_bytes) / payload;
        for (uint32_t i = 0; i < experts; ++i) { scores[i] += weight * (bank.scores ? static_cast<long double>(bank.scores[i]) : static_cast<long double>(bank.counts[i]) / bank.observations); }
    }
    output.swap(scores);
    return true;
} catch (...) { return false; }

bool ggml_moe_source_rank_statistics(const std::vector<ggml_moe_profile_bank_statistics> & banks, std::vector<int32_t> & ranks) try {
    std::vector<long double> scores;
    if (!ggml_moe_source_score_statistics(banks, scores)) { return false; }
    const auto experts = scores.size();
    std::vector<int32_t> result;
    result.reserve(experts);
    for (uint32_t i = 0; i < experts; ++i) { result.push_back(int32_t(i)); }
    std::sort(result.begin(), result.end(), [&](int32_t a, int32_t b) { return scores[a] > scores[b] || (scores[a] == scores[b] && a < b); });
    ranks.swap(result);
    return true;
} catch (...) { return false; }

bool ggml_moe_source_profile_initialize(ggml_moe_source_profile_learning & state, const int32_t * prior, uint32_t count, uint32_t n_experts) try {
    if (!prior || !count || count > n_experts || !n_experts || n_experts > (1u << 22)) { return false; }
    std::vector<int32_t> order(n_experts, -1);
    for (uint32_t i = 0; i < count; ++i) {
        if (prior[i] < 0 || uint32_t(prior[i]) >= n_experts || order[prior[i]] != -1) { return false; }
        order[prior[i]] = int32_t(i);
    }
    for (uint32_t i = 0; i < n_experts; ++i) { if (order[i] == -1) { order[i] = int32_t(count++); } }
    std::vector<uint64_t> counts(n_experts, 0);
    std::vector<double> heat(n_experts, 0);
    state.counts_.swap(counts);
    state.heat_.swap(heat);
    state.prior_.swap(order);
    state.observations_ = 0;
    return true;
} catch (...) { return false; }

bool ggml_moe_source_profile_observe(ggml_moe_source_profile_learning & state, const int32_t * routes, uint32_t count) {
    if (!routes || !count || state.counts_.empty() || count > UINT64_MAX - state.observations_) { return false; }
    for (uint32_t i = 0; i < count; ++i) {
        if (routes[i] < 0 || size_t(routes[i]) >= state.counts_.size()) { return false; }
    }
    // The total bounds each expert count, including repeated IDs in this window.
    for (uint32_t i = 0; i < count; ++i) { ++state.counts_[routes[i]]; }
    state.observations_ += count;
    return true;
}

bool ggml_moe_source_profile_accumulate(ggml_moe_source_profile_learning & state, const std::vector<float> & usage) {
    if (state.heat_.empty() || !state.observations_ || usage.size() != state.heat_.size()) { return false; }
    for (size_t i = 0; i < usage.size(); ++i) {
        if (!std::isfinite(usage[i]) || usage[i] < 0 || (!state.counts_[i] && usage[i] != 0) ||
                !std::isfinite(state.heat_[i] + double(usage[i]))) { return false; }
    }
    for (size_t i = 0; i < usage.size(); ++i) { state.heat_[i] += double(usage[i]); }
    return true;
}

bool ggml_moe_source_profile_snapshot(const ggml_moe_source_profile_learning & state, std::vector<uint64_t> & counts, std::vector<double> & heat,
        std::vector<int32_t> & ranks, uint64_t & observations, std::vector<int32_t> * prior) try {
    if (state.counts_.empty() || prior == &ranks) { return false; }
    auto new_counts = state.counts_;
    auto new_heat = state.heat_;
    std::vector<int32_t> new_prior;
    if (prior) { new_prior = state.prior_; }
    std::vector<int32_t> order;
    order.reserve(state.counts_.size());
    for (size_t i = 0; i < state.counts_.size(); ++i) { order.push_back(int32_t(i)); }
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return state.heat_[a] > state.heat_[b] || (state.heat_[a] == state.heat_[b] && state.prior_[a] < state.prior_[b]);
    });
    counts.swap(new_counts);
    heat.swap(new_heat);
    ranks.swap(order);
    if (prior) { prior->swap(new_prior); }
    observations = state.observations_;
    return true;
} catch (...) { return false; }

bool ggml_moe_source_profile_restore(ggml_moe_source_profile_learning & state, const std::vector<uint64_t> & counts,
        const std::vector<double> & heat, const std::vector<int32_t> & prior, uint64_t observations) try {
    if (counts.empty() || counts.size() > (1u << 22) || heat.size() != counts.size() || prior.size() != counts.size() ||
            !source_statistics_sum(counts.data(), uint32_t(counts.size()), observations)) { return false; }
    std::vector<uint8_t> seen(counts.size(), 0);
    double sum = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        if (!std::isfinite(heat[i]) || heat[i] < 0 || (!counts[i] && heat[i] != 0) ||
                prior[i] < 0 || size_t(prior[i]) >= counts.size() || seen[prior[i]]++) { return false; }
        sum += heat[i];
        if (!std::isfinite(sum)) { return false; }
    }
    auto new_counts = counts;
    auto new_heat = heat;
    auto new_prior = prior;
    state.counts_.swap(new_counts);
    state.heat_.swap(new_heat);
    state.prior_.swap(new_prior);
    state.observations_ = observations;
    return true;
} catch (...) { return false; }

bool ggml_moe_source_learning_valid(const ggml_backend_moe_source_learning_v1 * records, uint32_t count) try {
    if (!records || !count || count > GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS * GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS) { return false; }
    std::vector<ggml_backend_moe_source_statistics_v1> statistics;
    statistics.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const auto & record = records[i];
        if (record.struct_size != sizeof(record) || record.abi_version != 1 || !record.heat || !record.usage || !record.prior ||
                record.windows > record.source.observations || (record.windows == 0) != (record.source.observations == 0)) { return false; }
        statistics.push_back(record.source);
    }
    if (!ggml_moe_source_statistics_valid(statistics.data(), count)) { return false; }
    for (uint32_t i = 0; i < count; ++i) {
        const auto & record = records[i];
        const uint32_t experts = record.source.n_experts;
        ggml_moe_source_profile_learning state;
        if (!ggml_moe_source_profile_restore(state, {record.source.counts, record.source.counts + experts},
                {record.heat, record.heat + experts}, {record.prior, record.prior + experts}, record.source.observations)) { return false; }
        for (uint32_t expert = 0; expert < experts; ++expert) {
            if (!std::isfinite(record.usage[expert]) || record.usage[expert] < 0 || record.usage[expert] > float(record.source.counts[expert])) { return false; }
        }
    }
    return true;
} catch (...) { return false; }

bool ggml_moe_profile_plan_capacities(const std::vector<ggml_moe_profile_capacity_group> & groups,
        uint64_t budget, std::vector<uint32_t> & capacities, uint64_t & paid_bytes) try {
    if (groups.empty() || groups.size() > GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS) { return false; }
    struct candidate { uint32_t group; uint32_t rank; long double priority; };
    std::vector<candidate> candidates;
    std::vector<uint32_t> slots(groups.size());
    uint64_t paid = 0;
    size_t count = 0;
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto & group = groups[i];
        if (!group.per_slot_bytes || !group.minimum_slots || group.minimum_slots > group.priorities.size() ||
                group.priorities.size() > (1u << 22) - count || group.minimum_slots > (budget - paid) / group.per_slot_bytes) { return false; }
        paid += group.minimum_slots * group.per_slot_bytes;
        count += group.priorities.size();
        slots[i] = group.minimum_slots;
        for (size_t rank = 0; rank < group.priorities.size(); ++rank) {
            const auto score = group.priorities[rank];
            if (!std::isfinite(score) || score < 0 || (rank && score > group.priorities[rank - 1])) { return false; }
            if (rank >= group.minimum_slots) { candidates.push_back({uint32_t(i), uint32_t(rank), score}); }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const candidate & a, const candidate & b) {
        if (a.priority != b.priority) { return a.priority > b.priority; }
        return a.group != b.group ? a.group < b.group : a.rank < b.rank;
    });
    for (const auto & candidate : candidates) {
        const auto & group = groups[candidate.group];
        if (candidate.rank == slots[candidate.group] && group.per_slot_bytes <= budget - paid) {
            ++slots[candidate.group];
            paid += group.per_slot_bytes;
        }
    }
    capacities.swap(slots);
    paid_bytes = paid;
    return true;
} catch (...) { return false; }

static bool source_body_span(const ggml_tensor * tensor, size_t & span) {
    if (!tensor || tensor->type < 0 || tensor->type >= GGML_TYPE_COUNT ||
            tensor->op < 0 || tensor->op >= GGML_OP_COUNT) { return false; }
    const auto block = ggml_blck_size(tensor->type);
    const auto bytes = ggml_type_size(tensor->type);
    if (block <= 0 || !bytes || tensor->ne[0] <= 0 || tensor->ne[0] % block) { return false; }
    span = bytes;
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (tensor->ne[dim] <= 0 || uint64_t(tensor->ne[dim]) > SIZE_MAX) { return false; }
        const auto count = size_t(tensor->ne[dim]) / (dim ? 1 : size_t(block)) - 1;
        if (count && tensor->nb[dim] > (SIZE_MAX - span) / count) { return false; }
        span += count * tensor->nb[dim];
    }
    return true;
}

std::unique_ptr<ggml_moe_source_body> ggml_moe_source_body::prepare(const ggml_backend_moe_cpu_region_query_v1 & q) try {
    if (q.struct_size < sizeof(q) || !q.graph || !q.body_nodes || !q.n_body_nodes ||
            q.graph->n_nodes <= 0 || q.graph->n_nodes > q.graph->size || uint32_t(q.graph->n_nodes) != q.n_body_nodes || !q.graph->nodes ||
            !q.dynamic_inputs || !q.n_dynamic_inputs || !q.live_outputs || !q.n_live_outputs ||
            (q.n_sources && !q.sources) || !q.source_generation) { return nullptr; }
    const uint64_t count = uint64_t(q.n_dynamic_inputs) + q.n_sources + q.n_body_nodes;
    if (count > PTRDIFF_MAX / sizeof(ggml_tensor)) { return nullptr; }
    auto result = std::unique_ptr<ggml_moe_source_body>(new ggml_moe_source_body());
    result->tensors_.resize(size_t(count));
    std::vector<const ggml_tensor *> originals;
    originals.reserve(size_t(count));
    std::unordered_map<const ggml_tensor *, size_t> indices;
    const auto append = [&](const ggml_tensor * tensor) {
        size_t span;
        if (!source_body_span(tensor, span) || !indices.emplace(tensor, originals.size()).second) { return false; }
        originals.push_back(tensor);
        return true;
    };
    for (uint32_t i = 0; i < q.n_dynamic_inputs; ++i) {
        if (!append(q.dynamic_inputs[i])) { return nullptr; }
        result->dynamic_inputs_.push_back(&result->tensors_[originals.size() - 1]);
        if (q.dynamic_inputs[i] == q.ids) { result->ids_input_ = i; }
    }
    for (uint32_t i = 0; i < q.n_sources; ++i) {
        const auto & source = q.sources[i];
        size_t span;
        if (!source.witness || !source.data || source.generation != q.source_generation ||
                source.bytes > SIZE_MAX || !source_body_span(source.tensor, span) || source.bytes < span ||
                source.tensor->op != GGML_OP_NONE || source.tensor->view_src || !append(source.tensor)) { return nullptr; }
        for (const auto * src : source.tensor->src) { if (src) { return nullptr; } }
        auto clone = source;
        clone.tensor = &result->tensors_[originals.size() - 1];
        result->sources_.push_back(clone);
    }
    const size_t first_node = originals.size();
    for (uint32_t i = 0; i < q.n_body_nodes; ++i) {
        if (q.graph->nodes[i] != q.body_nodes[i] || !append(q.body_nodes[i])) { return nullptr; }
        result->nodes_.push_back(&result->tensors_[originals.size() - 1]);
    }
    const auto resolve = [&](const ggml_tensor * tensor, size_t consumer) -> ggml_tensor * {
        const auto found = indices.find(tensor);
        if (found == indices.end() || (found->second >= first_node && found->second >= consumer)) { return nullptr; }
        return &result->tensors_[found->second];
    };
    for (size_t i = 0; i < originals.size(); ++i) {
        const auto * original = originals[i];
        auto & clone = result->tensors_[i];
        clone = *original;
        clone.buffer = nullptr;
        clone.data = nullptr;
        clone.extra = nullptr;
        std::fill(std::begin(clone.src), std::end(clone.src), nullptr);
        clone.view_src = nullptr;
        if (i < first_node) {
            clone.op = GGML_OP_NONE;
            memset(clone.op_params, 0, sizeof(clone.op_params));
            clone.view_offs = 0;
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (original->src[s] && !(clone.src[s] = resolve(original->src[s], i))) { return nullptr; }
        }
        if (original->view_src) {
            size_t view_span, source_span;
            clone.view_src = resolve(original->view_src, i);
            if (!clone.view_src || !source_body_span(original, view_span) ||
                    !source_body_span(original->view_src, source_span) || original->view_offs > source_span ||
                    view_span > source_span - original->view_offs) { return nullptr; }
        } else if (original->view_offs) { return nullptr; }
    }
    for (uint32_t i = 0; i < q.n_live_outputs; ++i) {
        const auto found = indices.find(q.live_outputs[i]);
        if (found == indices.end() || found->second < first_node) { return nullptr; }
        const auto * clone = &result->tensors_[found->second];
        if (std::find(result->live_outputs_.begin(), result->live_outputs_.end(), clone) != result->live_outputs_.end()) { return nullptr; }
        result->live_outputs_.push_back(clone);
    }
    return result;
} catch (const std::bad_alloc &) { return nullptr; }

std::unique_ptr<ggml_moe_source_body> ggml_moe_source_body::compact(uint32_t capacity) const try {
    if (compact_ || !capacity || capacity > INT_MAX || ids_input_ >= dynamic_inputs_.size()) { return nullptr; }
    const auto * ids = dynamic_inputs_[ids_input_];
    if (ids->type != GGML_TYPE_I32 || ids->ne[0] > UINT32_MAX || ids->ne[1] > UINT32_MAX ||
            ids->ne[2] != 1 || ids->ne[3] != 1) { return nullptr; }
    std::vector<ggml_tensor *> graph_nodes;
    for (const auto * node : nodes_) { graph_nodes.push_back(const_cast<ggml_tensor *>(node)); }
    ggml_cgraph graph = {};
    graph.nodes = graph_nodes.data(); graph.size = graph.n_nodes = graph_nodes.size();
    ggml_backend_moe_cpu_region_query_v1 query = {};
    query.struct_size = sizeof(query); query.graph = &graph;
    query.body_nodes = nodes_.data(); query.n_body_nodes = nodes_.size();
    query.dynamic_inputs = dynamic_inputs_.data(); query.n_dynamic_inputs = dynamic_inputs_.size(); query.ids = ids;
    query.live_outputs = live_outputs_.data(); query.n_live_outputs = live_outputs_.size();
    query.sources = sources_.data(); query.n_sources = sources_.size();
    query.source_generation = sources_.empty() ? 1 : sources_[0].generation;
    auto result = prepare(query);
    if (!result) { return nullptr; }
    result->row_capacity_ = ids->ne[1]; result->routes_per_row_ = ids->ne[0]; result->route_capacity_ = capacity; result->compact_ = true;
    result->input_layouts_.resize(dynamic_inputs_.size());
    result->route_dimensions_.resize(result->tensors_.size());
    const auto contiguous = [](ggml_tensor & tensor) {
        const auto block = ggml_blck_size(tensor.type);
        tensor.nb[0] = ggml_type_size(tensor.type);
        if (block <= 0 || !tensor.nb[0] || tensor.ne[0] <= 0 || tensor.ne[0] % block ||
                uint64_t(tensor.ne[0] / block) > SIZE_MAX / tensor.nb[0]) { return false; }
        tensor.nb[1] = ggml_row_size(tensor.type, tensor.ne[0]);
        for (int d = 2; d < GGML_MAX_DIMS; ++d) {
            if (tensor.ne[d - 1] <= 0 || uint64_t(tensor.ne[d - 1]) > SIZE_MAX / tensor.nb[d - 1]) { return false; }
            tensor.nb[d] = tensor.nb[d - 1] * tensor.ne[d - 1];
        }
        size_t span;
        return source_body_span(&tensor, span);
    };
    for (size_t i = 0; i < dynamic_inputs_.size(); ++i) {
        auto & target = result->tensors_[i];
        const auto * input = dynamic_inputs_[i];
        if (i == ids_input_) { target.ne[0] = 1; target.ne[1] = capacity; result->route_dimensions_[i] = 1; }
        else {
            if (input->type != GGML_TYPE_F32 || input->nb[0] != sizeof(float) || input->ne[3] != 1 ||
                    ids->ne[0] % input->ne[1] || (input->ne[2] != 1 && input->ne[2] != ids->ne[1]) ||
                    uint64_t(input->ne[0]) > SIZE_MAX / sizeof(float)) { return nullptr; }
            result->input_layouts_[i] = {input->ne[2], input->ne[1], input->nb[2], input->nb[1], size_t(input->ne[0]) * sizeof(float)};
            target.ne[1] = 1; target.ne[2] = capacity;
            result->route_dimensions_[i] = 2;
        }
        if (!contiguous(target)) { return nullptr; }
    }
    const auto route_dimension = [&](const ggml_tensor * tensor) {
        return result->route_dimensions_[tensor - result->tensors_.data()];
    };
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto * original = nodes_[i];
        const size_t index = dynamic_inputs_.size() + sources_.size() + i;
        auto & node = result->tensors_[index];
        const auto * a = node.src[0];
        const auto * b = node.src[1];
        if (!a) { return nullptr; }
        switch (node.op) {
            case GGML_OP_MUL_MAT_ID:
                if (!b || node.src[2] != result->dynamic_inputs_[ids_input_] || a->ne[0] != b->ne[0] || b->ne[1] != 1) { return nullptr; }
                node.ne[0] = a->ne[1]; node.ne[1] = 1; node.ne[2] = b->ne[2]; node.ne[3] = 1;
                result->route_dimensions_[index] = 2;
                break;
            case GGML_OP_ADD_ID:
            case GGML_OP_ADD:
            case GGML_OP_MUL:
            case GGML_OP_DIV:
            case GGML_OP_GLU:
            case GGML_OP_UNARY:
            case GGML_OP_SQR:
            case GGML_OP_SCALE:
            case GGML_OP_CLAMP:
            case GGML_OP_CONT:
                for (int d = 1; d < GGML_MAX_DIMS; ++d) { node.ne[d] = a->ne[d]; }
                result->route_dimensions_[index] = route_dimension(a);
                if (node.op == GGML_OP_ADD || node.op == GGML_OP_MUL || node.op == GGML_OP_DIV) {
                    if (!b) { return nullptr; }
                    for (int d = 0; d < GGML_MAX_DIMS; ++d) { if (node.ne[d] % b->ne[d]) { return nullptr; } }
                }
                break;
            case GGML_OP_GET_ROWS:
                if (b != result->dynamic_inputs_[ids_input_] || a->ne[2] != b->ne[1] || a->ne[3] != b->ne[2]) { return nullptr; }
                node.ne[0] = a->ne[0]; node.ne[1] = b->ne[0]; node.ne[2] = b->ne[1]; node.ne[3] = b->ne[2];
                result->route_dimensions_[index] = 2;
                break;
            case GGML_OP_REPEAT: {
                bool expert_gather = false;
                for (const auto * consumer : nodes_) {
                    expert_gather |= consumer->op == GGML_OP_GET_ROWS && consumer->src[0] == original && consumer->src[1] == ids;
                }
                if (expert_gather) { node.ne[2] = capacity; }
                else {
                    node.ne[1] = 1; node.ne[2] = capacity;
                }
                for (int d = 0; d < GGML_MAX_DIMS; ++d) { if (node.ne[d] % a->ne[d]) { return nullptr; } }
                result->route_dimensions_[index] = 2;
                break;
            }
            case GGML_OP_RESHAPE:
                if (route_dimension(a)) {
                    node.ne[1] = 1; node.ne[2] = capacity;
                    uint64_t elements = 1, source_elements = 1;
                    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                        if (uint64_t(node.ne[d]) > UINT64_MAX / elements || uint64_t(a->ne[d]) > UINT64_MAX / source_elements) { return nullptr; }
                        elements *= node.ne[d]; source_elements *= a->ne[d];
                    }
                    if (elements != source_elements) { return nullptr; }
                    result->route_dimensions_[index] = 2;
                }
                break;
            case GGML_OP_VIEW:
                if (original->ne[1] != original->src[0]->ne[1] || original->ne[2] != original->src[0]->ne[2] ||
                        original->ne[3] != original->src[0]->ne[3]) { return nullptr; }
                for (int d = 1; d < GGML_MAX_DIMS; ++d) { node.ne[d] = a->ne[d]; node.nb[d] = a->nb[d]; }
                result->route_dimensions_[index] = route_dimension(a);
                break;
            default:
                return nullptr;
        }
        if (node.op != GGML_OP_VIEW && !contiguous(node)) { return nullptr; }
        if (node.view_src) {
            size_t span, parent_span;
            if (!source_body_span(&node, span) || !source_body_span(node.view_src, parent_span) ||
                    node.view_offs > parent_span || span > parent_span - node.view_offs) { return nullptr; }
        }
    }
    return result;
} catch (const std::bad_alloc &) { return nullptr; }

bool ggml_moe_source_body::bind_routes(uint32_t count) {
    if (!compact_ || !count || count > route_capacity_) { return false; }
    for (size_t i = 0; i < tensors_.size(); ++i) {
        const auto dimension = route_dimensions_[i];
        if (!dimension) { continue; }
        auto & tensor = tensors_[i];
        tensor.ne[dimension] = count;
        for (int d = dimension + 1; d < GGML_MAX_DIMS; ++d) { tensor.nb[d] = tensor.nb[d - 1] * tensor.ne[d - 1]; }
    }
    return true;
}

bool ggml_moe_source_body::gather_input(size_t input, const void * source, size_t source_bytes,
        const uint32_t * rows, const uint32_t * routes, uint32_t count, void * destination, size_t destination_bytes) const {
    if (!compact_ || input >= input_layouts_.size() || input == ids_input_ || !source || !destination ||
            !rows || !routes || !count || count > uint64_t(dynamic_inputs_[input]->ne[2])) { return false; }
    const auto & layout = input_layouts_[input];
    if (count > SIZE_MAX / layout.route_bytes || count * layout.route_bytes > destination_bytes) { return false; }
    const auto destination_address = reinterpret_cast<uintptr_t>(destination);
    const size_t written = count * layout.route_bytes;
    if (written > UINTPTR_MAX - destination_address) { return false; }
    const auto overlaps = [&](const void * data, size_t bytes) {
        const auto address = reinterpret_cast<uintptr_t>(data);
        return bytes > UINTPTR_MAX - address ||
            (address < destination_address + written && destination_address < address + bytes);
    };
    if (overlaps(source, source_bytes) || overlaps(rows, size_t(count) * sizeof(*rows)) ||
            overlaps(routes, size_t(count) * sizeof(*routes))) { return false; }
    const auto offset = [&](uint32_t i, size_t & result) {
        if (rows[i] >= row_capacity_ || routes[i] >= routes_per_row_) { return false; }
        const auto row = size_t(rows[i] % layout.rows), route = size_t(routes[i] % layout.routes);
        if ((row && layout.row_stride > SIZE_MAX / row) || (route && layout.route_stride > SIZE_MAX / route)) { return false; }
        result = row * layout.row_stride;
        const auto column = route * layout.route_stride;
        if (column > SIZE_MAX - result) { return false; }
        result += column;
        return result <= source_bytes && layout.route_bytes <= source_bytes - result;
    };
    for (uint32_t i = 0; i < count; ++i) { size_t at; if (!offset(i, at)) { return false; } }
    for (uint32_t i = 0; i < count; ++i) {
        size_t at; offset(i, at);
        memcpy(static_cast<uint8_t *>(destination) + i * layout.route_bytes, static_cast<const uint8_t *>(source) + at, layout.route_bytes);
    }
    return true;
}

size_t ggml_moe_source_operation_group_size(const std::vector<ggml_moe_source_operation> & operations, size_t index) {
    if (index >= operations.size()) { return 0; }
    const auto consecutive = [&](size_t count) {
        if (count > operations.size() - index || operations[index].original_index == SIZE_MAX) { return false; }
        for (size_t k = 1; k < count; ++k) {
            if (operations[index + k].original_index != operations[index].original_index + k) { return false; }
        }
        return true;
    };
    const auto * node = operations[index].tensor;
    if (node->op == GGML_OP_RMS_NORM && consecutive(2)) {
        const auto * multiply = operations[index + 1].tensor;
        if (multiply->op == GGML_OP_MUL && multiply->src[0] == node) { return 2; }
    }
    if (consecutive(9)) {
        const ggml_op expected[] = {GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL,
            GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_UNARY};
        bool match = true;
        for (size_t k = 0; k < 9; ++k) { match &= operations[index + k].tensor->op == expected[k]; }
        if (match) { return 9; }
    }
    return 1;
}

bool ggml_moe_source_expert_prepare(const ggml_backend_moe_hybrid_region_v1 & input,
        const ggml_backend_moe_cpu_prepared_region_v1_t * cpu_regions, ggml_moe_source_expert & result) {
    const auto reject = [&](const char * stage) {
        fprintf(stderr, "moe-source-core-preparation: stage=expert_%s rows=%u routes=%u ids_pitch=%zu ids_width=%zu\n",
            stage, input.geometry.row_capacity, input.geometry.routes_per_row,
            input.ids ? input.ids->nb[1] : 0, size_t(input.geometry.routes_per_row) * sizeof(int32_t));
        return false;
    };
    if (!cpu_regions || ggml_backend_moe_hybrid_validate_buckets_v1(&input) != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK ||
            input.n_cpu_batch_queries != 1 || !input.query || input.query->n_live_outputs != 1 ||
            !input.query->sources || !input.query->live_outputs || !input.activation || !input.ids || !input.output) {
        return reject("descriptor");
    }
    const auto & q = *input.query;
    const auto * down = input.output;
    const auto * glu = down && down->op == GGML_OP_MUL_MAT_ID ? down->src[1] : nullptr;
    const bool routed_operation = q.flags == GGML_BACKEND_MOE_CPU_REGION_FLAG_V1_ROUTED_OPERATION;
    const bool gated = glu && glu->op == GGML_OP_GLU &&
        (ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU || ggml_get_glu_op(glu) == GGML_GLU_OP_GEGLU);
    const bool squared_relu = glu && glu->op == GGML_OP_SQR && glu->src[0] &&
        glu->src[0]->op == GGML_OP_UNARY && ggml_get_unary_op(glu->src[0]) == GGML_UNARY_OP_RELU;
    if (!routed_operation && !gated && !squared_relu) { return reject("activation"); }
    ggml_moe_source_expert pending;
    pending.region = input;
    pending.region.query = nullptr;
    pending.region.cpu_queries = nullptr;
    pending.region.cpu_batch_queries = nullptr;
    pending.region.body_query = nullptr;
    pending.banks.resize(q.n_sources);
    for (uint32_t b = 0; b < q.n_sources; ++b) {
        const auto & source = q.sources[b];
        if (!source.tensor || !source.witness || !source.data || source.bytes == 0 || source.expert_stride == 0 ||
                source.expert_stride > SIZE_MAX || source.bytes > SIZE_MAX || source.tensor->ne[2] <= 0 ||
                uint64_t(source.tensor->ne[2]) != input.geometry.expert_count || source.tensor->ne[3] != 1) { return reject("bank_descriptor"); }
        auto & bank = pending.banks[b];
        bank.source = source;
        bank.metadata = *source.tensor;
        bank.expert_bytes = ggml_nbytes(source.tensor);
        if (bank.expert_bytes != source.bytes || bank.metadata.nb[2] != source.expert_stride ||
                uint64_t(input.geometry.expert_count - 1) > (source.bytes - 1) / source.expert_stride) { return reject("bank_extent"); }
        bank.expert_bytes = size_t(source.expert_stride);
        if (bank.expert_bytes > source.bytes - size_t(input.geometry.expert_count - 1) * source.expert_stride) { return reject("bank_stride"); }
    }
    if (routed_operation) {
        if (q.n_sources != 1 || !cpu_regions[0] || q.activation->ne[0] <= 0 || q.activation->ne[0] > INT_MAX ||
                q.live_outputs[0]->ne[0] <= 0 || q.live_outputs[0]->ne[0] > INT_MAX) { return reject("projection"); }
        pending.routed_operation = true;
        pending.roles[0] = {0, 0, q.sources[0].tensor->nb[1], q.sources[0].tensor->type};
        pending.input_width = q.activation->ne[0]; pending.output_width = q.live_outputs[0]->ne[0];
        pending.cpu_graph_uid = q.graph_uid; pending.cpu_graph_generation = q.graph_generation;
        pending.cpu_source_generation = q.source_generation; pending.cpu_region = cpu_regions[0];
        pending.body = ggml_moe_source_body::prepare(q);
        if (!pending.body) { return reject("projection_body"); }
        result = std::move(pending);
        for (auto & bank : result.banks) { bank.source.tensor = &bank.metadata; }
        return true;
    }
    const ggml_tensor * values[] = {gated ? glu->src[0] : glu->src[0]->src[0],
        gated ? glu->src[1] : glu->src[0]->src[0], down};
    pending.generic_body = !gated || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU;
    uint32_t widths[3] = {}, heights[3] = {};
    for (uint32_t role = 0; role < 3; ++role) {
        const auto * value = values[role];
        if (!value || value->ne[0] <= 0 || value->ne[0] > INT_MAX) { return reject("role_shape"); }
        heights[role] = uint32_t(value->ne[0]);
        size_t row_offset = 0;
        if (value->op == GGML_OP_VIEW) {
            if (role == 2 || value->view_src != value->src[0] || value->view_offs % sizeof(float) || value->nb[0] != sizeof(float)) { return reject("role_view"); }
            row_offset = value->view_offs / sizeof(float);
            value = value->view_src;
        }
        if (!value || value->op != GGML_OP_MUL_MAT_ID || value->src[2] != input.ids ||
                (role < 2 && value->src[1] != input.activation)) { return reject("role_node"); }
        const auto * weight = value->src[0];
        uint32_t bank = 0;
        while (bank < q.n_sources && q.sources[bank].tensor != weight && q.sources[bank].witness != weight) { ++bank; }
        if (bank == q.n_sources || !weight || weight->ne[0] <= 0 || weight->ne[0] > INT_MAX || weight->nb[1] == 0 ||
                row_offset > uint64_t(weight->ne[1]) || heights[role] > uint64_t(weight->ne[1]) - row_offset ||
                row_offset > SIZE_MAX / weight->nb[1] ||
                (row_offset + heights[role] - 1) > SIZE_MAX / weight->nb[1]) { return reject("role_bank"); }
        const size_t last = (row_offset + heights[role] - 1) * weight->nb[1];
        const size_t row_bytes = ggml_row_size(weight->type, weight->ne[0]);
        if (weight->nb[1] < row_bytes || last > pending.banks[bank].expert_bytes ||
                row_bytes > pending.banks[bank].expert_bytes - last) { return reject("role_span"); }
        widths[role] = uint32_t(weight->ne[0]);
        pending.roles[role] = {bank, row_offset * weight->nb[1], weight->nb[1], weight->type};
    }
    if (widths[0] != widths[1] || heights[0] != heights[1] ||
            widths[2] != heights[0] || input.activation->type != GGML_TYPE_F32 || input.activation->nb[0] != sizeof(float) ||
            input.activation->nb[2] != size_t(widths[0]) * sizeof(float) || input.output->type != GGML_TYPE_F32 ||
            input.output->nb[1] != size_t(heights[2]) * sizeof(float) || input.ids->type != GGML_TYPE_I32 ||
            input.ids->nb[0] != sizeof(int32_t) || input.ids->nb[1] < size_t(input.geometry.routes_per_row) * sizeof(int32_t) ||
            input.ids->nb[1] % sizeof(int32_t) != 0 ||
            input.output->nb[2] != size_t(heights[2]) * input.geometry.routes_per_row * sizeof(float)) { return reject("layout"); }
    pending.generic_body |= pending.roles[0].type != pending.roles[1].type;
    const size_t ids_width = size_t(input.geometry.routes_per_row) * sizeof(int32_t);
    const size_t ids_rows = input.geometry.row_capacity;
    if (!ids_rows || (ids_rows - 1) > (SIZE_MAX - ids_width) / input.ids->nb[1] ||
            (ids_rows - 1) * input.ids->nb[1] + ids_width > ggml_nbytes(input.ids)) { return reject("ids_span"); }
    const auto * batch = input.cpu_batch_queries[0];
    if (!batch || !cpu_regions[input.n_cpu_queries]) { return reject("cpu_batch"); }
    pending.body = ggml_moe_source_body::prepare(input.body_query ? *input.body_query : *batch);
    if (!pending.body) { return reject("body_graph"); }
    pending.input_width = widths[0];
    pending.hidden_width = heights[0];
    pending.output_width = heights[2];
    pending.cpu_graph_uid = batch->graph_uid;
    pending.cpu_graph_generation = batch->graph_generation;
    pending.cpu_source_generation = batch->source_generation;
    pending.cpu_region = cpu_regions[input.n_cpu_queries];
    result = std::move(pending);
    for (auto & bank : result.banks) { bank.source.tensor = &bank.metadata; }
    return true;
}

namespace {

template<bool ByteComparable>
struct source_tensor_witness {
    using storage = std::conditional_t<ByteComparable,
        std::array<unsigned char, sizeof(ggml_tensor::type) + ggml_moe_source_tensor_metadata_bytes>, ggml_tensor>;
    alignas(alignof(ggml_tensor)) storage value;

    explicit source_tensor_witness(const ggml_tensor & tensor) {
        if constexpr (ByteComparable) {
            memcpy(value.data(), &tensor.type, sizeof(tensor.type));
            memcpy(value.data() + sizeof(tensor.type), reinterpret_cast<const char *>(&tensor) + offsetof(ggml_tensor, buffer), ggml_moe_source_tensor_metadata_bytes);
        } else {
            value = tensor;
        }
    }

    bool matches(const ggml_tensor & tensor) const {
        if constexpr (ByteComparable) {
            ggml_type type;
            memcpy(&type, value.data(), sizeof(type));
            return tensor.type == type && !memcmp(reinterpret_cast<const char *>(&tensor) + offsetof(ggml_tensor, buffer),
                value.data() + sizeof(type), ggml_moe_source_tensor_metadata_bytes);
        } else {
            return ggml_moe_source_tensor_matches(tensor, value);
        }
    }
};

} // namespace

struct ggml_moe_source_program::impl {
    const ggml_cgraph * source = nullptr;
    uint64_t graph_uid = 0;
    int source_node_count = 0;
    ggml_graph_execution_certificate certificate = {};
    std::vector<const ggml_tensor *> originals;
    std::vector<const ggml_tensor *> validation_tensors;
    std::unordered_map<const ggml_tensor *, size_t> original_indices;
    std::vector<source_tensor_witness<ggml_moe_source_tensor_byte_comparable>> witnesses;
    std::vector<ggml_tensor> tensors;
    std::vector<uint8_t> omitted, body_storage;
    std::vector<ggml_tensor *> allocation_nodes, leafs;
    std::vector<const ggml_tensor *> outputs;
    std::vector<ggml_moe_source_public_binding> public_bindings;
    std::vector<ggml_moe_source_layer> layers;
    std::vector<ggml_moe_source_operation> epilogue;
    ggml_context * dependency_context = nullptr;
    ggml_cgraph allocation_graph = {};
    ggml_gallocr_t allocator = nullptr;
    size_t bytes = 0, operations = 0, owner_writes = 0, overlap_operations = 0, overlap_layers = 0;
    bool allocated = false;

    ~impl() { ggml_gallocr_free(allocator); ggml_free(dependency_context); }

    size_t index(const ggml_tensor * tensor) const {
        const auto it = original_indices.find(tensor);
        return it == original_indices.end() ? SIZE_MAX : it->second;
    }

    size_t root(size_t i) const {
        for (size_t depth = 0; i < originals.size() && depth < originals.size(); ++depth) {
            if (!originals[i]->view_src) { return i; }
            i = index(originals[i]->view_src);
        }
        return SIZE_MAX;
    }

    ggml_moe_source_operation operation(size_t i) {
        auto & tensor = tensors[i];
        const bool metadata = ggml_is_empty(&tensor) || tensor.op == GGML_OP_NONE || tensor.op == GGML_OP_VIEW || tensor.op == GGML_OP_RESHAPE ||
            tensor.op == GGML_OP_PERMUTE || tensor.op == GGML_OP_TRANSPOSE || !(tensor.flags & GGML_TENSOR_FLAG_COMPUTE);
        const auto effect = metadata ? ggml_moe_source_effect::metadata :
            (tensor.op == GGML_OP_CPY || tensor.op == GGML_OP_SET_ROWS || tensor.op == GGML_OP_SET) ?
            ggml_moe_source_effect::owner_write : ggml_moe_source_effect::private_value;
        operations += !metadata;
        owner_writes += effect == ggml_moe_source_effect::owner_write;
        return {&tensor, originals[i], effect, i};
    }

    bool overlap_safe(size_t i) const {
        const auto & tensor = tensors[i];
        const size_t base = root(i);
        if (base == SIZE_MAX || omitted[base] || (tensor.flags & GGML_TENSOR_FLAG_OUTPUT) ||
                (tensors[base].flags & GGML_TENSOR_FLAG_OUTPUT)) { return false; }
        return ggml_op_is_empty(tensor.op) || (ggml_op_is_pure(tensor.op) && !tensor.view_src && base == i);
    }

    bool schedule_overlap() {
        const size_t count = size_t(source_node_count);
        std::vector<uint8_t> moved(count, 0);
        for (size_t layer = 0; layer < layers.size(); ++layer) {
            const auto & region = layers[layer].expert->region;
            auto & tail = layer + 1 < layers.size() ? layers[layer + 1].prelude : epilogue;
            std::vector<uint8_t> available(originals.size(), 0);
            for (size_t i = 0; i < available.size(); ++i) {
                available[i] = !omitted[i] && (i < region.first_node || i >= count);
            }
            for (size_t cursor = 0; cursor < tail.size();) {
                const size_t unit = ggml_moe_source_operation_group_size(tail, cursor);
                bool safe = true, ready = true;
                for (size_t k = 0; k < unit; ++k) {
                    const size_t i = tail[cursor + k].original_index;
                    safe &= overlap_safe(i);
                    const auto internal = [&](size_t s) {
                        for (size_t j = 0; j < k; ++j) { if (tail[cursor + j].original_index == s) { return true; } }
                        return false;
                    };
                    for (const auto * input : tensors[i].src) {
                        if (!input) { continue; }
                        const size_t s = size_t(input - tensors.data());
                        ready &= available[s] || internal(s);
                    }
                    if (tensors[i].view_src) {
                        const size_t s = size_t(tensors[i].view_src - tensors.data());
                        ready &= available[s] || internal(s);
                    }
                }
                // Do not cross an unknown effect, owner write, public root or in-place value.
                if (!safe) { break; }
                if (ready) {
                    for (size_t k = 0; k < unit; ++k) {
                        const auto & op = tail[cursor + k];
                        layers[layer].overlap.push_back(op);
                        moved[op.original_index] = available[op.original_index] = 1;
                        overlap_operations += op.effect != ggml_moe_source_effect::metadata;
                    }
                }
                cursor += unit;
            }
            overlap_layers += !layers[layer].overlap.empty();
            tail.erase(std::remove_if(tail.begin(), tail.end(), [&](const ggml_moe_source_operation & op) {
                return moved[op.original_index] != 0;
            }), tail.end());
        }
        std::vector<size_t> positions(count, SIZE_MAX);
        allocation_nodes.clear();
        const auto append = [&](const std::vector<ggml_moe_source_operation> & values) {
            for (const auto & op : values) {
                if (op.original_index >= count || positions[op.original_index] != SIZE_MAX) { return false; }
                positions[op.original_index] = allocation_nodes.size();
                allocation_nodes.push_back(op.tensor);
            }
            return true;
        };
        for (const auto & layer : layers) {
            if (!append(layer.prelude) || !append(layer.overlap)) { return false; }
            for (size_t n = layer.expert->region.first_node; n < layer.expert->region.last_node; ++n) {
                if (!body_storage[n]) { continue; }
                if (positions[n] != SIZE_MAX) { return false; }
                positions[n] = allocation_nodes.size();
                allocation_nodes.push_back(&tensors[n]);
            }
            const size_t i = layer.expert->region.last_node;
            if (positions[i] != SIZE_MAX) { return false; }
            positions[i] = allocation_nodes.size();
            allocation_nodes.push_back(&tensors[i]);
        }
        if (!append(epilogue)) { return false; }
        for (size_t i = 0; i < count; ++i) {
            if (omitted[i] && !body_storage[i]) { continue; }
            if (positions[i] == SIZE_MAX) { return false; }
            const auto before = [&](const ggml_tensor * value) {
                if (!value) { return true; }
                const size_t s = size_t(value - tensors.data());
                return (!omitted[s] || body_storage[s]) && (s >= count || positions[s] < positions[i] || (s == i && tensors[i].op == GGML_OP_CPY));
            };
            for (const auto * input : tensors[i].src) { if (!before(input)) { return false; } }
            if (!before(tensors[i].view_src)) { return false; }
        }
        return true;
    }

    bool add_allocation_dependencies(const std::vector<ggml_moe_source_allocation_dependency> & dependencies) {
        if (dependencies.empty()) { return true; }
        if (dependencies.size() > size_t(INT_MAX) - allocation_nodes.size() ||
                dependencies.size() > SIZE_MAX / ggml_tensor_overhead()) { return false; }
        std::vector<size_t> positions(tensors.size(), SIZE_MAX);
        for (size_t i = 0; i < allocation_nodes.size(); ++i) { positions[size_t(allocation_nodes[i] - tensors.data())] = i; }
        std::vector<std::vector<ggml_tensor *>> keep(allocation_nodes.size());
        for (const auto & dependency : dependencies) {
            const size_t value = index(dependency.tensor), until = index(dependency.until);
            if (value == SIZE_MAX || until >= size_t(source_node_count) || omitted[value] || omitted[until] ||
                    positions[until] == SIZE_MAX ||
                    (value < size_t(source_node_count) && positions[value] >= positions[until])) { return false; }
            auto & inputs = keep[positions[until]];
            if (std::find(inputs.begin(), inputs.end(), &tensors[value]) == inputs.end()) { inputs.push_back(&tensors[value]); }
        }
        dependency_context = ggml_init({dependencies.size() * ggml_tensor_overhead(), nullptr, true});
        if (!dependency_context) { return false; }
        std::vector<ggml_tensor *> nodes;
        nodes.reserve(allocation_nodes.size() + dependencies.size());
        for (size_t i = 0; i < allocation_nodes.size(); ++i) {
            nodes.push_back(allocation_nodes[i]);
            for (size_t k = 0; k < keep[i].size(); k += GGML_MAX_SRC) {
                // Match the scheduler's allocation-only views; replay uses the original operation lists.
                auto * dep = ggml_view_tensor(dependency_context, keep[i][k]);
                for (size_t s = 0; s < GGML_MAX_SRC && k + s < keep[i].size(); ++s) { dep->src[s] = keep[i][k + s]; }
                nodes.push_back(dep);
            }
        }
        allocation_nodes = std::move(nodes);
        return true;
    }

    bool prepare(const ggml_cgraph * graph, const std::vector<const ggml_moe_source_expert *> & experts,
            ggml_backend_buffer_type_t buft, const ggml_moe_source_schedule_options & options) {
        const auto reject = [](const char * stage, const ggml_tensor * tensor = nullptr) {
            GGML_LOG_ERROR("moe-source-program-preparation: stage=%s name=%s op=%s ne=%lld,%lld,%lld,%lld\n",
                stage, tensor ? tensor->name : "none", tensor ? ggml_op_name(tensor->op) : "none",
                tensor ? (long long) tensor->ne[0] : 0, tensor ? (long long) tensor->ne[1] : 0,
                tensor ? (long long) tensor->ne[2] : 0, tensor ? (long long) tensor->ne[3] : 0);
            return false;
        };
        if (!graph || graph->n_nodes <= 0 || !graph->nodes || !graph->uid || experts.empty() || !buft) { return reject("graph"); }
        source = graph;
        graph_uid = graph->uid;
        source_node_count = graph->n_nodes;
        certificate = graph->execution_certificate;
        if (certificate.magic != GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC || certificate.n_rows == 0 ||
                certificate.source_graph_uid == 0 || certificate.split_graph_uid != graph_uid) { return reject("certificate"); }
        const size_t count = graph->n_nodes;
        originals.reserve(count);
        original_indices.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            if (!graph->nodes[i] || !original_indices.emplace(graph->nodes[i], originals.size()).second) { return reject("node_identity", graph->nodes[i]); }
            originals.push_back(graph->nodes[i]);
        }
        for (size_t i = 0; i < originals.size(); ++i) {
            const auto add = [&](const ggml_tensor * t) {
                if (t && original_indices.emplace(t, originals.size()).second) { originals.push_back(t); }
            };
            for (const auto * input : originals[i]->src) { add(input); }
            add(originals[i]->view_src);
        }
        if (originals.size() > INT_MAX) { return reject("metadata_capacity"); }
        witnesses.reserve(originals.size());
        tensors.resize(originals.size());
        omitted.resize(originals.size());
        body_storage.resize(originals.size());
        std::vector<uint8_t> exported(count);
        if (graph->use_counts && graph->visited_hash_set.size &&
                graph->visited_hash_set.used && graph->visited_hash_set.keys) {
            std::vector<int32_t> uses(count);
            for (size_t i = 0; i < count; ++i) {
                for (const auto * input : originals[i]->src) {
                    const size_t s = index(input);
                    if (s < count) {
                        if (uses[s] == INT32_MAX) { return reject("use_count_capacity", originals[s]); }
                        ++uses[s];
                    }
                }
            }
            // Graph views share parent use counts. Keep values read outside this split.
            for (size_t i = 0; i < count; ++i) {
                const size_t slot = ggml_hash_find(&graph->visited_hash_set, originals[i]);
                if (slot == GGML_HASHSET_FULL || !ggml_bitset_get(graph->visited_hash_set.used, slot) ||
                        graph->use_counts[slot] < uses[i]) { return reject("use_count_identity", originals[i]); }
                exported[i] = graph->use_counts[slot] > uses[i];
            }
        }
        size_t previous = 0;
        for (const auto * expert : experts) {
            if (!expert) { return reject("expert_identity"); }
            const auto & region = expert->region;
            if (region.first_node < previous || region.first_node > region.last_node || region.last_node >= count ||
                    region.output != graph->nodes[region.last_node] || region.output->view_src ||
                    !region.geometry.row_capacity || region.geometry.row_capacity > certificate.n_rows ||
                    region.source_graph_uid != certificate.source_graph_uid || region.split_graph_uid != graph_uid ||
                    region.owner_generation != certificate.owner_generation) { return reject("expert_boundary", region.output); }
            for (size_t i = region.first_node; i < region.last_node; ++i) {
                if ((originals[i]->flags & GGML_TENSOR_FLAG_OUTPUT) || exported[i]) { return reject("omitted_output", originals[i]); }
                omitted[i] = 1;
                body_storage[i] = expert->generic_body;
            }
            previous = size_t(region.last_node) + 1;
        }
        // Scan metadata in address order; execution keeps the original node order.
        validation_tensors = originals;
        std::sort(validation_tensors.begin(), validation_tensors.end(), std::less<const ggml_tensor *>{});
        for (const auto * tensor : validation_tensors) { witnesses.emplace_back(*tensor); }
        for (size_t i = 0; i < originals.size(); ++i) {
            tensors[i] = *originals[i];
        }
        for (size_t i = 0; i < originals.size(); ++i) {
            auto & tensor = tensors[i];
            const size_t base = root(i);
            if (base == SIZE_MAX || (!omitted[i] && omitted[base])) { return reject("view_root", originals[i]); }
            for (auto & input : tensor.src) {
                if (!input) { continue; }
                const size_t s = index(input);
                if (s == SIZE_MAX) { return reject("input_identity", originals[i]); }
                input = &tensors[s];
            }
            if (tensor.view_src) {
                size_t offset = 0;
                for (size_t v = i; v != base; v = index(originals[v]->view_src)) {
                    if (originals[v]->view_offs > SIZE_MAX - offset) { return reject("view_offset", originals[i]); }
                    offset += originals[v]->view_offs;
                }
                const size_t extent = ggml_nbytes(originals[base]);
                if (!ggml_is_empty(originals[i]) &&
                        (offset > extent || ggml_nbytes(originals[i]) > extent - offset)) { return reject("view_extent", originals[i]); }
                tensor.view_src = &tensors[base];
                tensor.view_offs = offset;
                tensor.data = nullptr;
                tensor.buffer = nullptr;
                tensor.extra = nullptr;
            } else if (i < count && tensor.op != GGML_OP_NONE) {
                tensor.data = nullptr;
                tensor.buffer = nullptr;
                tensor.extra = nullptr;
            } else if (!ggml_is_empty(&tensor) && !tensor.data) { return reject("leaf_storage", originals[i]); }

        }
        for (const auto * expert : experts) {
            if (expert->generic_body) {
                for (uint32_t i = expert->region.first_node; i <= expert->region.last_node; ++i) {
                    if (tensors[i].op == GGML_OP_MUL_MAT_ID) { tensors[i].src[0] = nullptr; }
                }
                auto & output = tensors[expert->region.last_node];
                const size_t activation = index(expert->region.activation), ids = index(expert->region.ids);
                if (activation == SIZE_MAX || ids == SIZE_MAX || omitted[activation] || omitted[ids]) { return reject("expert_input", expert->region.output); }
                output.op = GGML_OP_NONE;
                // Both GPU passes share scratch. Keep inputs alive and allocate their result before that scratch.
                output.flags |= GGML_TENSOR_FLAG_INPUT;
                output.src[3] = &tensors[activation];
                output.src[4] = &tensors[ids];
                continue;
            }
            auto & output = tensors[expert->region.last_node];
            output.op = GGML_OP_NONE;
            std::fill(std::begin(output.src), std::end(output.src), nullptr);
            size_t next = 0;
            for (const auto * input : {expert->region.activation, expert->region.ids}) {
                for (const auto i : {index(input), root(index(input))}) {
                    if (i == SIZE_MAX || omitted[i]) { return reject("expert_input", input); }
                    if (std::find(output.src, output.src + next, &tensors[i]) == output.src + next) { output.src[next++] = &tensors[i]; }
                }
            }
        }
        if (options.bind_leaf) {
            std::vector<uint8_t> reachable(tensors.size(), 0);
            std::vector<size_t> pending;
            for (size_t i = 0; i < count; ++i) { if (!omitted[i]) { pending.push_back(i); } }
            while (!pending.empty()) {
                const size_t i = pending.back(); pending.pop_back();
                if (reachable[i]) { continue; }
                reachable[i] = 1;
                const auto & tensor = tensors[i];
                if (i < count) {
                    for (const auto * input : tensor.src) {
                        if (input) { pending.push_back(size_t(input - tensors.data())); }
                    }
                }
                if (tensor.view_src) { pending.push_back(size_t(tensor.view_src - tensors.data())); }
            }
            for (size_t i = 0; i < tensors.size(); ++i) {
                if (!reachable[i]) { continue; }
                auto & tensor = tensors[i];
                if (!tensor.view_src && originals[i]->op == GGML_OP_NONE && tensor.op == GGML_OP_NONE && !ggml_is_empty(&tensor) &&
                        !(tensor.flags & GGML_TENSOR_FLAG_INPUT)) {
                    auto buffer = tensor.buffer;
                    auto * data = tensor.data;
                    if (!options.bind_leaf(options.leaf_context, originals[i], &buffer, &data) || !buffer || !data) {
                        return reject("leaf_binding", originals[i]);
                    }
                    size_t span;
                    const uintptr_t base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
                    const uintptr_t pointer = reinterpret_cast<uintptr_t>(data);
                    const size_t capacity = ggml_backend_buffer_get_size(buffer);
                    if (!source_body_span(&tensor, span) || !base || pointer < base || pointer - base > capacity ||
                            span > capacity - (pointer - base)) { return reject("leaf_binding_extent", originals[i]); }
                    if (tensor.buffer != buffer || tensor.data != data) { tensor.extra = nullptr; }
                    tensor.buffer = buffer;
                    tensor.data = data;
                }
            }
        }
        for (size_t i = 0; i < count; ++i) {
            if (omitted[i] && !body_storage[i]) { continue; }
            auto & tensor = tensors[i];
            for (const auto * input : tensor.src) {
                if (!input) { continue; }
                const size_t s = size_t(input - tensors.data());
                if ((omitted[s] && !body_storage[s]) || (s < count && s >= i && !(s == i && tensor.op == GGML_OP_CPY))) { return reject("node_dependency", originals[i]); }
            }
            if ((originals[i]->flags & GGML_TENSOR_FLAG_OUTPUT) || exported[i]) {
                const size_t base = root(i);
                if (base == SIZE_MAX || omitted[base]) { return reject("output_root", originals[i]); }
                if (exported[i]) { tensor.flags |= GGML_TENSOR_FLAG_OUTPUT; }
                tensors[base].flags |= GGML_TENSOR_FLAG_OUTPUT;
                const auto * output = originals[(originals[i]->flags & GGML_TENSOR_FLAG_OUTPUT) ? i : base];
                if (std::find(outputs.begin(), outputs.end(), output) == outputs.end()) { outputs.push_back(output); }
            }
            allocation_nodes.push_back(&tensor);
        }
        if (outputs.empty()) { return reject("public_outputs"); }
        public_bindings.reserve(outputs.size());
        for (const auto * output : outputs) {
            public_bindings.push_back({output, output->buffer, output->data, ggml_nbytes(output)});
        }
        for (size_t i = count; i < tensors.size(); ++i) { leafs.push_back(&tensors[i]); }
        size_t cursor = 0;
        layers.reserve(experts.size());
        for (const auto * expert : experts) {
            ggml_moe_source_layer layer;
            layer.expert = expert;
            if (index(expert->region.activation) == SIZE_MAX || index(expert->region.ids) == SIZE_MAX) { return reject("layer_inputs", expert->region.output); }
            layer.activation = &tensors[index(expert->region.activation)];
            layer.ids = &tensors[index(expert->region.ids)];
            layer.output = &tensors[expert->region.last_node];
            while (cursor < expert->region.first_node) { layer.prelude.push_back(operation(cursor++)); }
            cursor = size_t(expert->region.last_node) + 1;
            layers.push_back(std::move(layer));
        }
        while (cursor < count) { epilogue.push_back(operation(cursor++)); }
        if (options.overlap_independent_ordinary && !schedule_overlap()) { return reject("overlap_schedule"); }
        if (!add_allocation_dependencies(options.allocation_dependencies)) { return reject("allocation_dependencies"); }
        allocation_graph.n_nodes = int(allocation_nodes.size());
        allocation_graph.nodes = allocation_nodes.data();
        allocation_graph.n_leafs = int(leafs.size());
        allocation_graph.leafs = leafs.data();
        allocator = ggml_gallocr_new(buft);
        if (!allocator) { return reject("allocator"); }
        ggml_gallocr_reserve_n_size(allocator, &allocation_graph, nullptr, nullptr, &bytes);
        return ggml_gallocr_get_metadata_size(allocator) != SIZE_MAX;
    }
};

ggml_moe_source_program::ggml_moe_source_program() = default;
ggml_moe_source_program::~ggml_moe_source_program() = default;

bool ggml_moe_source_program::prepare(const ggml_cgraph * graph,
        const std::vector<const ggml_moe_source_expert *> & experts, ggml_backend_buffer_type_t buft) {
    return prepare(graph, experts, buft, {});
}

bool ggml_moe_source_program::prepare(const ggml_cgraph * graph,
        const std::vector<const ggml_moe_source_expert *> & experts, ggml_backend_buffer_type_t buft,
        const ggml_moe_source_schedule_options & options) {
    auto pending = std::make_unique<impl>();
    if (!pending->prepare(graph, experts, buft, options)) { return false; }
    state = std::move(pending);
    return true;
}

bool ggml_moe_source_program::allocate() {
    if (!state || state->allocated || !ggml_gallocr_reserve(state->allocator, &state->allocation_graph) ||
            ggml_gallocr_get_buffer_size(state->allocator, 0) != state->bytes ||
            !ggml_gallocr_alloc_graph(state->allocator, &state->allocation_graph)) { return false; }
    state->allocated = true;
    return true;
}

bool ggml_moe_source_program::matches(const ggml_cgraph * graph) const {
    return graph && matches(graph, graph->execution_certificate);
}

bool ggml_moe_source_program::matches(const ggml_cgraph * graph, const ggml_graph_execution_certificate & certificate) const {
    if (!state || !graph || graph != state->source || graph->uid != state->graph_uid ||
            graph->n_nodes != state->source_node_count || !graph->nodes ||
            memcmp(&certificate, &state->certificate, sizeof(state->certificate))) { return false; }
    if constexpr (std::has_unique_object_representations<ggml_tensor *>::value &&
            std::has_unique_object_representations<const ggml_tensor *>::value) {
        if (memcmp(graph->nodes, state->originals.data(), size_t(state->source_node_count) * sizeof(*graph->nodes))) { return false; }
    } else {
        for (int i = 0; i < state->source_node_count; ++i) { if (graph->nodes[i] != state->originals[i]) { return false; } }
    }
    for (size_t i = 0; i < state->validation_tensors.size(); ++i) {
        if (!state->witnesses[i].matches(*state->validation_tensors[i])) { return false; }
    }
    return true;
}

const ggml_tensor * ggml_moe_source_program::original_node(size_t index) const {
    return state && index < size_t(state->source_node_count) ? state->originals[index] : nullptr;
}

ggml_tensor * ggml_moe_source_program::find(const ggml_tensor * original) const {
    if (!state) { return nullptr; }
    const size_t i = state->index(original);
    return i < state->tensors.size() && (!state->omitted[i] || state->body_storage[i]) ? &state->tensors[i] : nullptr;
}

bool ggml_moe_source_program::closed_cut(const ggml_tensor * const * cut, size_t count,
        const ggml_tensor * const * retained, size_t retained_count) const {
    if (!state || !cut || !count || !retained || !retained_count) { return false; }
    const auto contains = [](const ggml_tensor * const * values, size_t n, const ggml_tensor * value) {
        return std::find(values, values + n, value) != values + n;
    };
    for (size_t k = 0; k < count; ++k) {
        const auto * value = cut[k];
        const auto found = std::find_if(state->tensors.begin(), state->tensors.end(),
            [&](const ggml_tensor & tensor) { return &tensor == value; });
        if (!value || found == state->tensors.end()) { return false; }
        if (contains(retained, retained_count, value)) { continue; }
        const size_t index = size_t(found - state->tensors.begin());
        const auto * original = state->originals[index];
        if (state->tensors[index].flags & GGML_TENSOR_FLAG_OUTPUT) { return false; }
        for (size_t j = 0; j < state->originals.size(); ++j) {
            const auto * consumer = state->originals[j];
            bool reads = consumer->view_src == original;
            for (const auto * input : consumer->src) { reads |= input == original; }
            if (reads && !contains(cut, count, &state->tensors[j])) { return false; }
        }
    }
    return true;
}

const std::vector<ggml_moe_source_layer> & ggml_moe_source_program::layers() const { return state->layers; }
const std::vector<ggml_moe_source_operation> & ggml_moe_source_program::epilogue() const { return state->epilogue; }
const std::vector<const ggml_tensor *> & ggml_moe_source_program::public_outputs() const { return state->outputs; }
const std::vector<ggml_moe_source_public_binding> & ggml_moe_source_program::public_bindings() const { return state->public_bindings; }
const ggml_graph_execution_certificate & ggml_moe_source_program::certificate() const { return state->certificate; }
size_t ggml_moe_source_program::storage_bytes() const { return state ? state->bytes : 0; }
size_t ggml_moe_source_program::operation_count() const { return state ? state->operations : 0; }
size_t ggml_moe_source_program::owner_write_count() const { return state ? state->owner_writes : 0; }
size_t ggml_moe_source_program::overlap_operation_count() const { return state ? state->overlap_operations : 0; }
size_t ggml_moe_source_program::overlap_layer_count() const { return state ? state->overlap_layers : 0; }
