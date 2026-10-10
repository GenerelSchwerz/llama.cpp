#include "llama-context.h"
#include "llama-draft-vocab.h"
#include "ggml-moe-source-program.h"
#include "llama-staged-input.h"
#include "../ggml/src/moe-fidelity-config.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-memory-hybrid.h"
#include "llama-kv-cache.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "llama-ext.h"
#include "llama-sampler.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

//
// llama_context
//

std::vector<std::vector<int32_t>> llama_moe_profile_parse(
        const uint8_t * data, size_t bytes, const std::vector<uint32_t> & expert_counts) {
    const auto reject = [](const char * reason) { throw std::runtime_error(std::string("MoE STRP profile: ") + reason); };
    if (!data || bytes < 24 || bytes > 64 * 1024 * 1024 || memcmp(data, "STRP", 4)) { reject("invalid extent or magic"); }
    const auto u32 = [&](size_t offset) {
        return uint32_t(data[offset]) | uint32_t(data[offset + 1]) << 8 |
            uint32_t(data[offset + 2]) << 16 | uint32_t(data[offset + 3]) << 24;
    };
    const uint32_t layers = u32(8), experts = u32(12), slots = u32(16), pairs = u32(20);
    if (u32(4) != 1 || layers == 0 || layers > 65536 || layers != expert_counts.size() ||
            experts == 0 || experts > 65536 || !pairs || slots > pairs) { reject("unsupported header or layer geometry"); }
    uint64_t total = 0;
    uint32_t maximum = 0;
    for (const auto count : expert_counts) {
        total += count;
        maximum = std::max(maximum, count);
    }
    if (maximum != experts || total > (1u << 22) || pairs > total) { reject("expert geometry or pair count mismatch"); }
    const uint64_t pair_end = 24ull + 4ull * pairs;
    const uint64_t table_bytes = 4ull * layers * experts;
    if (bytes != pair_end && bytes != pair_end + table_bytes) { reject("truncated file or unknown trailing data"); }
    std::vector<std::vector<int32_t>> ranks(layers), indices(layers);
    for (uint32_t layer = 0; layer < layers; ++layer) { indices[layer].assign(expert_counts[layer], -1); }
    for (uint32_t i = 0; i < pairs; ++i) {
        const size_t offset = 24 + size_t(i) * 4;
        const uint32_t layer = uint32_t(data[offset]) | uint32_t(data[offset + 1]) << 8;
        const uint32_t expert = uint32_t(data[offset + 2]) | uint32_t(data[offset + 3]) << 8;
        if (layer >= layers || expert >= expert_counts[layer] || indices[layer][expert] != -1) { reject("invalid or duplicate pair"); }
        indices[layer][expert] = i;
        ranks[layer].push_back(expert);
    }
    if (bytes != pair_end) {
        for (uint32_t layer = 0; layer < layers; ++layer) {
            for (uint32_t expert = 0; expert < experts; ++expert) {
                const uint32_t expected = expert < expert_counts[layer] ? uint32_t(indices[layer][expert]) : UINT32_MAX;
                if (u32(size_t(pair_end) + (size_t(layer) * experts + expert) * 4) != expected) { reject("invalid historical slot-index table"); }
            }
        }
    }
    return ranks;
}

llama_moe_profile_statistics llama_moe_profile_statistics_parse(
        const uint8_t * data, size_t bytes, const std::vector<llama_moe_source_group> & sources) {
    const auto reject = [](const char * reason) { throw std::runtime_error(std::string("MoE source statistics: ") + reason); };
    if (!data || bytes < 24 || bytes > 256 * 1024 * 1024 || memcmp(data, "GGUF", 4)) { reject("invalid extent or magic"); }
    const auto u64 = [&](size_t offset, uint32_t width = 8) {
        uint64_t result = 0;
        for (uint32_t i = 0; i < width; ++i) { result |= uint64_t(data[offset + i]) << (8 * i); }
        return result;
    };
    if (u64(4, 4) != 3 || u64(8) != 0 || u64(16) > 32) { reject("unsupported GGUF metadata header"); }
    gguf_context_ptr metadata(gguf_init_from_buffer(data, bytes, {true, nullptr}));
    if (!metadata) { reject("invalid metadata"); }
    const size_t end = gguf_get_data_offset(metadata.get()), alignment = gguf_get_alignment(metadata.get());
    if (end > bytes || alignment > 64 * 1024 * 1024) { reject("invalid metadata extent or alignment"); }
    const size_t padding = (alignment - end % alignment) % alignment;
    if (end != bytes && (bytes - end != padding || !std::all_of(data + end, data + bytes, [](uint8_t value) { return value == 0; }))) {
        reject("invalid metadata padding or trailing data");
    }
    const auto key = [&](const char * name, gguf_type type) {
        const auto index = gguf_find_key(metadata.get(), name);
        if (index < 0 || gguf_get_kv_type(metadata.get(), index) != type) { reject("missing or mistyped field"); }
        return index;
    };
    const auto array = [&](const char * name, gguf_type type, size_t count) {
        const auto index = key(name, GGUF_TYPE_ARRAY);
        if (gguf_get_arr_type(metadata.get(), index) != type || gguf_get_arr_n(metadata.get(), index) != count) { reject("array type or length mismatch"); }
        return gguf_get_arr_data(metadata.get(), index);
    };
    const auto version = gguf_get_val_u32(metadata.get(), key("moe.profile.version", GGUF_TYPE_UINT32));
    if (version != 3 && bytes > 64 * 1024 * 1024) { reject("calibration metadata exceeds file capacity"); }
    if (version != 1 && version != 2 && version != 3) { reject("unsupported statistics version"); }
    if (version == 1 && gguf_find_key(metadata.get(), "moe.profile.ranking_scores") >= 0) { reject("scores require statistics version2"); }
    const auto domain_key = key("moe.profile.source_domains", GGUF_TYPE_ARRAY);
    const size_t n_sources = gguf_get_arr_n(metadata.get(), domain_key);
    if (!n_sources || n_sources > GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS * GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS || n_sources > SIZE_MAX / 4) {
        reject("source count exceeds shared capacity");
    }
    const auto * domains = static_cast<const uint32_t *>(array("moe.profile.source_domains", GGUF_TYPE_UINT32, n_sources));
    const auto * types = static_cast<const uint32_t *>(array("moe.profile.source_types", GGUF_TYPE_UINT32, n_sources));
    const auto * shapes = static_cast<const int64_t *>(array("moe.profile.source_shapes", GGUF_TYPE_INT64, n_sources * 4));
    const auto * observations = static_cast<const uint64_t *>(array("moe.profile.observations", GGUF_TYPE_UINT64, n_sources));
    const auto * offsets = static_cast<const uint64_t *>(array("moe.profile.expert_offsets", GGUF_TYPE_UINT64, n_sources + 1));
    const auto * name_offsets = static_cast<const uint64_t *>(array("moe.profile.source_name_offsets", GGUF_TYPE_UINT64, n_sources + 1));
    const auto names_key = key("moe.profile.source_name_bytes", GGUF_TYPE_ARRAY);
    const auto counts_key = key("moe.profile.expert_counts", GGUF_TYPE_ARRAY);
    const auto provenance_key = key("moe.profile.provenance", GGUF_TYPE_ARRAY);
    const auto n_names = gguf_get_arr_n(metadata.get(), names_key);
    const auto n_counts = gguf_get_arr_n(metadata.get(), counts_key);
    const auto n_provenance = gguf_get_arr_n(metadata.get(), provenance_key);
    if (!n_names || n_counts > (1u << 22) || !n_provenance || n_provenance > 4096) { reject("data count exceeds statistics capacity"); }
    const auto * names = static_cast<const uint8_t *>(array("moe.profile.source_name_bytes", GGUF_TYPE_UINT8, n_names));
    const auto * counts = static_cast<const uint64_t *>(array("moe.profile.expert_counts", GGUF_TYPE_UINT64, n_counts));
    const auto * provenance = static_cast<const uint8_t *>(array("moe.profile.provenance", GGUF_TYPE_UINT8, n_provenance));
    const auto * scores = version >= 2 ? static_cast<const double *>(array("moe.profile.ranking_scores", GGUF_TYPE_FLOAT64, n_counts)) : nullptr;
    const double * heat = nullptr;
    const float * usage = nullptr;
    const int32_t * prior = nullptr;
    const uint64_t * windows = nullptr;
    if (version == 3) {
        if (gguf_get_val_u32(metadata.get(), key("moe.profile.learning_policy", GGUF_TYPE_UINT32)) != 1) { reject("unsupported learning policy"); }
        heat = static_cast<const double *>(array("moe.profile.learned_heat", GGUF_TYPE_FLOAT64, n_counts));
        usage = static_cast<const float *>(array("moe.profile.decayed_usage", GGUF_TYPE_FLOAT32, n_counts));
        prior = static_cast<const int32_t *>(array("moe.profile.prior_indices", GGUF_TYPE_INT32, n_counts));
        windows = static_cast<const uint64_t *>(array("moe.profile.learning_windows", GGUF_TYPE_UINT64, n_sources));
    } else {
        for (const auto * name : {"moe.profile.learning_policy", "moe.profile.learned_heat", "moe.profile.decayed_usage", "moe.profile.prior_indices", "moe.profile.learning_windows"}) {
            if (gguf_find_key(metadata.get(), name) >= 0) { reject("learning state requires statistics version3"); }
        }
    }
    if (name_offsets[0] || name_offsets[n_sources] != n_names || offsets[0] || offsets[n_sources] != n_counts ||
            memchr(provenance, 0, n_provenance)) { reject("invalid offsets or provenance"); }

    llama_moe_profile_statistics result;
    result.provenance.assign(reinterpret_cast<const char *>(provenance), n_provenance);
    std::map<uint32_t, std::unordered_map<const ggml_tensor *, size_t>> source_indices;
    std::unordered_map<std::string, std::map<uint32_t, size_t>> name_indices;
    uint64_t model_experts = 0;
    for (const auto & source : sources) {
        for (const auto & bank : source.banks) {
            if (bank.status != GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) { continue; }
            if (!bank.tensor || bank.names.empty() || bank.tensor->ne[2] <= 0 || bank.tensor->ne[2] > INT32_MAX) { reject("source identity or expert geometry unavailable"); }
            auto & indices = source_indices[source.domain];
            const auto inserted = indices.emplace(bank.tensor, result.sources.size());
            if (inserted.second) {
                if (uint64_t(bank.tensor->ne[2]) > (1u << 22) - model_experts) { reject("model expert count exceeds statistics capacity"); }
                model_experts += bank.tensor->ne[2];
                result.sources.push_back({bank.tensor, source.domain, 0, {}});
            }
            const size_t index = inserted.first->second;
            for (const auto & name : bank.names) {
                if (name.empty() || name.size() > 4096 || name.find('\0') != std::string::npos) { reject("invalid loader source name"); }
                auto & aliases = name_indices[name];
                const auto alias = aliases.emplace(source.domain, index);
                if (!alias.second && alias.first->second != index) { alias.first->second = SIZE_MAX; }
            }
        }
    }
    if (result.sources.empty()) { reject("model has no returned expert sources"); }
    for (size_t i = 0; i < n_sources; ++i) {
        if (name_offsets[i] >= name_offsets[i + 1] || name_offsets[i + 1] > n_names || name_offsets[i + 1] - name_offsets[i] > 4096 ||
                offsets[i] > offsets[i + 1] || offsets[i + 1] > n_counts) { reject("invalid source extent"); }
        const auto * begin = names + size_t(name_offsets[i]);
        const auto name_size = size_t(name_offsets[i + 1] - name_offsets[i]);
        if (memchr(begin, 0, name_size)) { reject("embedded source name terminator"); }
        const std::string name(reinterpret_cast<const char *>(begin), name_size);
        const auto named = name_indices.find(name);
        if (named == name_indices.end()) { reject("unknown loader source name"); }
        const auto identity = named->second.find(domains[i]);
        if (identity == named->second.end() || identity->second == SIZE_MAX) { reject("unknown or ambiguous source domain"); }
        auto & bound = result.sources[identity->second];
        const auto * tensor = bound.tensor;
        if (types[i] != uint32_t(tensor->type) || offsets[i + 1] - offsets[i] != uint64_t(tensor->ne[2]) ||
                !std::equal(tensor->ne, tensor->ne + GGML_MAX_DIMS, shapes + i * 4)) { reject("source type or shape mismatch"); }
        uint64_t total = 0;
        for (uint64_t index = offsets[i]; index < offsets[i + 1]; ++index) {
            if (counts[index] > UINT64_MAX - total) { reject("observation count overflow"); }
            total += counts[index];
        }
        if (total != observations[i]) { reject("observation total mismatch"); }
        double score_sum = 0;
        for (uint64_t index = offsets[i]; scores && index < offsets[i + 1]; ++index) {
            if (!std::isfinite(scores[index]) || scores[index] < 0 || (version != 3 && !total && scores[index] != 0)) { reject("invalid ranking score"); }
            score_sum += scores[index];
            if (!std::isfinite(score_sum)) { reject("ranking score overflow"); }
        }
        const auto * first = counts + size_t(offsets[i]);
        const auto * last = counts + size_t(offsets[i + 1]);
        if (!bound.counts.empty()) {
            if (heat && (bound.windows != windows[i] || !std::equal(bound.heat.begin(), bound.heat.end(), heat + size_t(offsets[i])) ||
                    !std::equal(bound.usage.begin(), bound.usage.end(), usage + size_t(offsets[i])) ||
                    !std::equal(bound.prior.begin(), bound.prior.end(), prior + size_t(offsets[i])))) { reject("contradictory learned aliases"); }
            if (bound.observations != total || !std::equal(bound.counts.begin(), bound.counts.end(), first) ||
                    (scores && !std::equal(bound.scores.begin(), bound.scores.end(), scores + size_t(offsets[i])))) { reject("contradictory source aliases"); }
        } else {
            bound.observations = total;
            bound.counts.assign(first, last);
            if (scores) { bound.scores.assign(scores + size_t(offsets[i]), scores + size_t(offsets[i + 1])); }
            if (heat) {
                bound.heat.assign(heat + size_t(offsets[i]), heat + size_t(offsets[i + 1]));
                bound.usage.assign(usage + size_t(offsets[i]), usage + size_t(offsets[i + 1]));
                bound.prior.assign(prior + size_t(offsets[i]), prior + size_t(offsets[i + 1]));
                bound.windows = windows[i];
            }
        }
    }
    for (const auto & source : result.sources) {
        if (source.counts.empty()) { reject("profile omits a returned expert source"); }
        if (version == 3) {
            ggml_moe_source_profile_learning state;
            const ggml_backend_moe_source_learning_v1 record{sizeof(ggml_backend_moe_source_learning_v1), 1,
                {source.tensor, source.counts.data(), source.observations, uint32_t(source.counts.size()), source.domain},
                source.heat.data(), source.usage.data(), source.prior.data(), source.windows};
            if (!ggml_moe_source_learning_valid(&record, 1) ||
                    !ggml_moe_source_profile_restore(state, source.counts, source.heat, source.prior, source.observations)) { reject("invalid learned state or cadence"); }
            std::vector<uint64_t> counts;
            std::vector<double> heat;
            std::vector<int32_t> ranks;
            uint64_t observations = 0;
            if (!ggml_moe_source_profile_snapshot(state, counts, heat, ranks, observations)) { reject("cannot rank learned state"); }
            for (size_t i = 0; i < ranks.size(); ++i) {
                const double score = double(ranks.size() - i) / ranks.size();
                if (source.scores[ranks[i]] != score) { reject("learned ranking score mismatch"); }
            }
        }
    }
    return result;
}

std::vector<uint8_t> llama_moe_profile_statistics_serialize(const llama_moe_profile_statistics & statistics, const std::vector<llama_moe_source_group> & sources) {
    const auto * measured = &statistics;
    gguf_context_ptr metadata(gguf_init_empty());
    std::vector<uint8_t> names;
    std::vector<uint64_t> name_offsets{0}, offsets{0}, observations, counts;
    std::vector<double> scores, heat;
    std::vector<float> usage;
    std::vector<int32_t> prior;
    std::vector<uint64_t> windows;
    const bool learned = !statistics.sources.empty() && !statistics.sources.front().heat.empty();
    const bool scored = measured && !measured->sources.empty() && !measured->sources.front().scores.empty();
    std::vector<uint32_t> domains, types;
    std::vector<int64_t> shapes;
    std::map<uint32_t, std::unordered_set<const ggml_tensor *>> seen;
    for (const auto & source : sources) {
        for (const auto & bank : source.banks) {
            if (bank.status != GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE || !seen[source.domain].insert(bank.tensor).second) { continue; }
            if (!bank.tensor || bank.names.empty() || bank.tensor->ne[2] <= 0 || counts.size() > (1u << 22) || uint64_t(bank.tensor->ne[2]) > (1u << 22) - counts.size() || domains.size() >= GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS * GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS) { throw std::runtime_error("invalid statistics geometry"); }
            const auto & name = bank.names.front();
            if (name.empty() || name.size() > 4096 || name.find('\0') != std::string::npos) { throw std::runtime_error("invalid statistics source name"); }
            names.insert(names.end(), name.begin(), name.end());
            name_offsets.push_back(names.size());
            domains.push_back(source.domain);
            types.push_back(uint32_t(bank.tensor->type));
            shapes.insert(shapes.end(), bank.tensor->ne, bank.tensor->ne + GGML_MAX_DIMS);
            uint64_t total = 0;
            const llama_moe_profile_source_statistics * actual = nullptr;
            if (measured) {
                const auto found = std::find_if(measured->sources.begin(), measured->sources.end(), [&](const auto & value) { return value.tensor == bank.tensor && value.domain == source.domain; });
                if (found == measured->sources.end() || found->counts.size() != size_t(bank.tensor->ne[2])) { throw std::runtime_error("missing measured source statistics"); }
                actual = &*found;
                if (scored ? actual->scores.size() != actual->counts.size() : !actual->scores.empty()) { throw std::runtime_error("incomplete ranking scores"); }
                if (scored) { scores.insert(scores.end(), actual->scores.begin(), actual->scores.end()); }
                if (learned) {
                    if (!scored || actual->heat.size() != actual->counts.size() || actual->usage.size() != actual->counts.size() || actual->prior.size() != actual->counts.size()) { throw std::runtime_error("incomplete learned state"); }
                    heat.insert(heat.end(), actual->heat.begin(), actual->heat.end());
                    usage.insert(usage.end(), actual->usage.begin(), actual->usage.end());
                    prior.insert(prior.end(), actual->prior.begin(), actual->prior.end());
                    windows.push_back(actual->windows);
                } else if (!actual->heat.empty() || !actual->usage.empty() || !actual->prior.empty() || actual->windows) { throw std::runtime_error("mixed learned and calibration statistics"); }
            }
            for (int64_t i = 0; i < bank.tensor->ne[2]; ++i) {
                const uint64_t value = actual->counts[i];
                if (value > UINT64_MAX - total) { throw std::runtime_error("statistics occurrence overflow"); }
                counts.push_back(value); total += value;
            }
            if (actual && total != actual->observations) { throw std::runtime_error("statistics observation mismatch"); }
            offsets.push_back(counts.size());
            observations.push_back(total);
        }
    }
    if (domains.size() != statistics.sources.size()) { throw std::runtime_error("unknown or duplicate statistics source"); }
    const std::string & provenance = statistics.provenance;
    if (provenance.empty() || provenance.size() > 4096 || provenance.find('\0') != std::string::npos) { throw std::runtime_error("invalid statistics provenance"); }
    gguf_set_val_u32(metadata.get(), "moe.profile.version", learned ? 3 : scored ? 2 : 1);
    if (learned) {
        gguf_set_val_u32(metadata.get(), "moe.profile.learning_policy", 1);
        gguf_set_arr_data(metadata.get(), "moe.profile.learned_heat", GGUF_TYPE_FLOAT64, heat.data(), heat.size());
        gguf_set_arr_data(metadata.get(), "moe.profile.decayed_usage", GGUF_TYPE_FLOAT32, usage.data(), usage.size());
        gguf_set_arr_data(metadata.get(), "moe.profile.prior_indices", GGUF_TYPE_INT32, prior.data(), prior.size());
        gguf_set_arr_data(metadata.get(), "moe.profile.learning_windows", GGUF_TYPE_UINT64, windows.data(), windows.size());
    }
    if (scored) { gguf_set_arr_data(metadata.get(), "moe.profile.ranking_scores", GGUF_TYPE_FLOAT64, scores.data(), scores.size()); }
    gguf_set_arr_data(metadata.get(), "moe.profile.source_name_bytes", GGUF_TYPE_UINT8, names.data(), names.size());
    gguf_set_arr_data(metadata.get(), "moe.profile.source_name_offsets", GGUF_TYPE_UINT64, name_offsets.data(), name_offsets.size());
    gguf_set_arr_data(metadata.get(), "moe.profile.source_domains", GGUF_TYPE_UINT32, domains.data(), domains.size());
    gguf_set_arr_data(metadata.get(), "moe.profile.source_types", GGUF_TYPE_UINT32, types.data(), types.size());
    gguf_set_arr_data(metadata.get(), "moe.profile.source_shapes", GGUF_TYPE_INT64, shapes.data(), shapes.size());
    gguf_set_arr_data(metadata.get(), "moe.profile.expert_offsets", GGUF_TYPE_UINT64, offsets.data(), offsets.size());
    gguf_set_arr_data(metadata.get(), "moe.profile.observations", GGUF_TYPE_UINT64, observations.data(), observations.size());
    gguf_set_arr_data(metadata.get(), "moe.profile.expert_counts", GGUF_TYPE_UINT64, counts.data(), counts.size());
    gguf_set_arr_data(metadata.get(), "moe.profile.provenance", GGUF_TYPE_UINT8, provenance.data(), provenance.size());
    const size_t bytes = gguf_get_meta_size(metadata.get());
    if (bytes > size_t(learned ? 256 : 64) * 1024 * 1024) { throw std::runtime_error("profile metadata exceeds file capacity"); }
    std::vector<uint8_t> result(bytes);
    gguf_get_meta_data(metadata.get(), result.data());
    (void) llama_moe_profile_statistics_parse(result.data(), result.size(), sources);
    return result;
}


llama_moe_profile_statistics llama_moe_profile_learning_baseline(const std::vector<llama_moe_source_group> & sources,
        const llama_moe_profile_statistics & initial, const std::vector<ggml_backend_moe_static_profile_v1> & profiles) {
    if (sources.size() > GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS * GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS ||
            profiles.size() > GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS || (!initial.sources.empty() && !profiles.empty())) {
        throw std::runtime_error("invalid full-model learning seed");
    }
    if (!initial.sources.empty()) { (void) llama_moe_profile_statistics_serialize(initial, sources); }
    std::unordered_map<const ggml_tensor *, const ggml_backend_moe_static_profile_v1 *> ranked;
    uint64_t entries = 0;
    for (const auto & profile : profiles) {
        if (!profile.down || !profile.experts || !profile.n_experts || profile.n_experts > (1u << 22) - entries ||
                !ranked.emplace(profile.down, &profile).second) { throw std::runtime_error("invalid ranked learning seed"); }
        entries += profile.n_experts;
    }
    llama_moe_profile_statistics result;
    result.provenance = "Full-model learning snapshot; raw online occurrences and accepted maintenance heat; unobserved sources retain original priors";
    std::map<uint32_t, std::unordered_map<const ggml_tensor *, size_t>> identities;
    std::unordered_set<const ggml_tensor *> used_profiles;
    entries = 0;
    for (const auto & group : sources) {
        std::vector<const llama_moe_source_bank *> banks;
        std::vector<ggml_moe_profile_bank_statistics> statistics;
        const ggml_backend_moe_static_profile_v1 * seed = nullptr;
        for (const auto & bank : group.banks) {
            if (bank.status != GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) { continue; }
            if (!bank.tensor || bank.tensor->ne[2] <= 0 || bank.tensor->ne[2] > (1u << 22) || !bank.tensor->nb[2]) { throw std::runtime_error("invalid learning source geometry"); }
            banks.push_back(&bank);
            const auto profile = ranked.find(bank.tensor);
            if (profile != ranked.end()) {
                if (profile->second->n_experts > uint64_t(bank.tensor->ne[2]) || (seed && (seed->n_experts != profile->second->n_experts ||
                        !std::equal(seed->experts, seed->experts + seed->n_experts, profile->second->experts)))) { throw std::runtime_error("conflicting group learning seeds"); }
                seed = profile->second; used_profiles.insert(bank.tensor);
            }
            if (!initial.sources.empty()) {
                const auto source = std::find_if(initial.sources.begin(), initial.sources.end(), [&](const auto & value) {
                    return value.tensor == bank.tensor && value.domain == group.domain;
                });
                if (source == initial.sources.end()) { throw std::runtime_error("missing initial learning source"); }
                statistics.push_back({source->counts.data(), source->observations, bank.tensor->nb[2], uint32_t(source->counts.size()),
                    source->scores.empty() ? nullptr : source->scores.data()});
            }
        }
        if (banks.empty()) { continue; }
        const auto experts = uint32_t(banks.front()->tensor->ne[2]);
        std::vector<int32_t> order;
        if (!statistics.empty() && !ggml_moe_source_rank_statistics(statistics, order)) { throw std::runtime_error("cannot rank initial learning sources"); }
        if (order.empty()) { for (uint32_t i = 0; i < experts; ++i) { order.push_back(int32_t(i)); } }
        ggml_moe_source_profile_learning zero;
        if (!ggml_moe_source_profile_initialize(zero, seed ? seed->experts : order.data(), seed ? seed->n_experts : experts, experts)) {
            throw std::runtime_error("invalid full-model learning prior");
        }
        std::vector<uint64_t> counts;
        std::vector<double> heat;
        std::vector<int32_t> ranks, prior;
        uint64_t observations = 0;
        if (!ggml_moe_source_profile_snapshot(zero, counts, heat, ranks, observations, &prior)) { throw std::runtime_error("cannot snapshot initial learning prior"); }
        for (const auto * bank : banks) {
            if (bank->tensor->ne[2] != experts) { throw std::runtime_error("inconsistent group learning geometry"); }
            llama_moe_profile_source_statistics item{bank->tensor, group.domain, 0, counts};
            item.heat = heat; item.usage.assign(experts, 0); item.prior = prior; item.scores.resize(experts);
            for (uint32_t i = 0; i < experts; ++i) { item.scores[ranks[i]] = double(experts - i) / experts; }
            const auto learned = std::find_if(initial.sources.begin(), initial.sources.end(), [&](const auto & value) {
                return value.tensor == bank->tensor && value.domain == group.domain && !value.heat.empty();
            });
            if (learned != initial.sources.end()) { item = *learned; }
            auto & indices = identities[group.domain];
            const auto existing = indices.find(bank->tensor);
            if (existing != indices.end()) {
                const auto & previous = result.sources[existing->second];
                if (previous.counts != item.counts || previous.scores != item.scores || previous.prior != item.prior || previous.heat != item.heat ||
                        previous.usage != item.usage || previous.windows != item.windows || previous.observations != item.observations) {
                    throw std::runtime_error("conflicting shared learning source");
                }
                continue;
            }
            if (experts > (1u << 22) - entries || result.sources.size() >= GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS * GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS) {
                throw std::runtime_error("full-model learning storage exceeded");
            }
            entries += experts;
            indices.emplace(bank->tensor, result.sources.size()); result.sources.push_back(std::move(item));
        }
    }
    if (used_profiles.size() != ranked.size() || result.sources.empty()) { throw std::runtime_error("unknown learning seed or empty source catalog"); }
    (void) llama_moe_profile_statistics_serialize(result, sources);
    return result;
}

bool llama_moe_profile_learning_overlay(llama_moe_profile_statistics & snapshot, std::vector<uint8_t> & covered,
        const ggml_backend_moe_source_learning_v1 * records, uint32_t count) {
    if (covered.size() != snapshot.sources.size()) { return false; }
    if (!count) { return true; }
    if (!ggml_moe_source_learning_valid(records, count)) { return false; }
    std::vector<size_t> destinations(count);
    for (uint32_t i = 0; i < count; ++i) {
        const auto & record = records[i];
        const auto found = std::find_if(snapshot.sources.begin(), snapshot.sources.end(), [&](const auto & source) {
            return source.tensor == record.source.tensor && source.domain == record.source.domain;
        });
        if (found == snapshot.sources.end() || found->counts.size() != record.source.n_experts) { return false; }
        const size_t index = size_t(found - snapshot.sources.begin());
        destinations[i] = index;
        if (covered[index] && (found->heat.size() != found->counts.size() || found->usage.size() != found->counts.size() ||
                found->prior.size() != found->counts.size() || found->observations != record.source.observations || found->windows != record.windows ||
                !std::equal(found->counts.begin(), found->counts.end(), record.source.counts) ||
                !std::equal(found->heat.begin(), found->heat.end(), record.heat) ||
                !std::equal(found->usage.begin(), found->usage.end(), record.usage) ||
                !std::equal(found->prior.begin(), found->prior.end(), record.prior))) { return false; }
    }
    for (uint32_t i = 0; i < count; ++i) {
        const auto & record = records[i];
        auto & source = snapshot.sources[destinations[i]];
        if (covered[destinations[i]]) { continue; }
        const uint32_t experts = record.source.n_experts;
        source.counts.assign(record.source.counts, record.source.counts + experts);
        source.heat.assign(record.heat, record.heat + experts);
        source.usage.assign(record.usage, record.usage + experts);
        source.prior.assign(record.prior, record.prior + experts);
        source.observations = record.source.observations;
        source.windows = record.windows;
        ggml_moe_source_profile_learning state;
        std::vector<uint64_t> counts;
        std::vector<double> heat;
        std::vector<int32_t> ranks;
        uint64_t observations = 0;
        if (!ggml_moe_source_profile_restore(state, source.counts, source.heat, source.prior, source.observations) ||
                !ggml_moe_source_profile_snapshot(state, counts, heat, ranks, observations)) { return false; }
        source.scores.resize(experts);
        for (uint32_t rank = 0; rank < experts; ++rank) { source.scores[ranks[rank]] = double(experts - rank) / experts; }
        covered[destinations[i]] = 1;
    }
    return true;
}

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_DRAFT  : return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

static bool moe_hybrid_required_for_context(const char * mode, llama_context_type context_type) {
    if (mode != nullptr && strcmp(mode, "required") != 0 && strcmp(mode, "off") != 0) {
        throw std::runtime_error("GGML_MOE_HYBRID must be off or required");
    }
    GGML_UNUSED(context_type);
    return mode != nullptr && strcmp(mode, "required") == 0;
}

static bool moe_hybrid_execution_supported(bool required, uint32_t domain, uint32_t row_semantics) {
    constexpr bool main_speculative_supported = true;
    return !required || domain != GGML_GRAPH_EXECUTION_DOMAIN_MAIN ||
        row_semantics != GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE || main_speculative_supported;
}

static uint64_t graph_execution_next_owner_namespace() {
    static std::atomic<uint64_t> next_namespace { 1 };
    const uint64_t result = next_namespace.fetch_add(1, std::memory_order_relaxed);
    GGML_ASSERT(result != 0);
    return result;
}

static bool backend_supports_required_grouped_execution(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto * fn = reg != nullptr ? reinterpret_cast<ggml_backend_required_grouped_execution_supported_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_REQUIRED_GROUPED_EXECUTION_SUPPORTED_PROC_NAME)) : nullptr;
    return fn != nullptr && fn(backend);
}

static bool is_moe_cached_tensor(const ggml_tensor * tensor) {
    if (!tensor || !tensor->buffer) {
        return false;
    }
    auto * buft = ggml_backend_buffer_get_type(tensor->buffer);
    auto * dev  = ggml_backend_buft_get_device(buft);
    auto * reg  = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto * fn   = reg ? reinterpret_cast<ggml_backend_moe_cache_is_buffer_type_t>(
                            ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_IS_BUFFER_TYPE_PROC_NAME)) :
                        nullptr;
    return fn && fn(buft);
}

static bool graph_supports_required_grouped_execution(ggml_backend_sched_t sched, ggml_cgraph * gf) {
    bool                               participating = false;
    std::unordered_set<ggml_backend_t> checked;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto * node = ggml_graph_node(gf, i);
        // Splitting can replace a cached bank with a transfer tensor. Check every MMID owner.
        if (node->op != GGML_OP_MUL_MAT_ID || ggml_is_empty(node)) {
            continue;
        }
        participating = true;
        auto * owner  = ggml_backend_sched_get_tensor_backend(sched, node);
        if (checked.insert(owner).second && !backend_supports_required_grouped_execution(owner)) {
            return false;
        }
    }
    return participating;
}

static uint32_t required_grouped_execution_flags(uint32_t cache_slots, bool backend_supported) {
    return cache_slots > 0 && backend_supported ?
        GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED : GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE;
}

static bool ubatch_has_independent_rows(const llama_ubatch & ubatch) {
    if (ubatch.n_tokens == 0 || ubatch.n_seq_tokens != 1 ||
            ubatch.n_seqs != ubatch.n_tokens || ubatch.n_seqs_unq != ubatch.n_tokens ||
            ubatch.n_seq_id == nullptr || ubatch.seq_id == nullptr || ubatch.seq_id_unq == nullptr) {
        return false;
    }

    for (uint32_t row = 0; row < ubatch.n_tokens; ++row) {
        if (ubatch.n_seq_id[row] != 1 || ubatch.seq_id[row] == nullptr) {
            return false;
        }
        const llama_seq_id row_id = ubatch.seq_id[row][0];
        bool listed = false;
        for (uint32_t seq = 0; seq < ubatch.n_seqs_unq; ++seq) {
            listed = listed || ubatch.seq_id_unq[seq] == row_id;
        }
        if (!listed) {
            return false;
        }
        for (uint32_t previous = 0; previous < row; ++previous) {
            if (ubatch.seq_id[previous][0] == row_id) {
                return false;
            }
        }
    }

    return true;
}

struct llama_graph_execution_intent {
    uint32_t domain = GGML_GRAPH_EXECUTION_DOMAIN_INVALID;
    uint32_t row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INVALID;
    uint32_t verification_span = 0;
};

static bool verification_position_matches(llama_pos first, uint32_t offset, llama_pos current) {
    if (offset > (uint64_t) std::numeric_limits<llama_pos>::max() ||
            first > std::numeric_limits<llama_pos>::max() - (llama_pos) offset) {
        return false;
    }
    return current == first + (llama_pos) offset;
}

static bool batch_has_independent_rows(const llama_batch & batch) {
    if (batch.n_tokens <= 0 || batch.pos == nullptr || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return false;
    }
    for (int32_t row = 0; row < batch.n_tokens; ++row) {
        if (batch.n_seq_id[row] != 1 || batch.seq_id[row] == nullptr || batch.seq_id[row][0] < 0 ||
                batch.seq_id[row][0] >= LLAMA_MAX_SEQ) {
            return false;
        }
        for (int32_t previous = 0; previous < row; ++previous) {
            if (batch.seq_id[previous][0] == batch.seq_id[row][0]) {
                return false;
            }
        }
    }
    return true;
}

static bool batch_has_sequential_spans(const llama_batch & batch) {
    if (batch.n_tokens <= 0 || batch.pos == nullptr || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return false;
    }
    std::array<uint8_t, LLAMA_MAX_SEQ> seen = {};
    llama_seq_id current_seq = -1;
    llama_pos current_pos = 0;
    for (int32_t row = 0; row < batch.n_tokens; ++row) {
        if (batch.n_seq_id[row] != 1 || batch.seq_id[row] == nullptr) {
            return false;
        }
        const llama_seq_id seq_id = batch.seq_id[row][0];
        if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ) {
            return false;
        }
        if (seq_id != current_seq) {
            if (seen[seq_id]) {
                return false;
            }
            seen[seq_id] = 1;
            current_seq = seq_id;
        } else if (!verification_position_matches(current_pos, 1, batch.pos[row])) {
            return false;
        }
        current_pos = batch.pos[row];
    }
    return true;
}

static uint32_t speculative_batch_row_semantics(const llama_batch & batch) {
    if (batch_has_independent_rows(batch)) {
        return GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT;
    }
    if (batch_has_sequential_spans(batch)) {
        return GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL;
    }
    return GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INVALID;
}

static uint32_t speculative_execution_domain(llama_context_type context_type) {
    switch (context_type) {
        case LLAMA_CONTEXT_TYPE_DRAFT: return GGML_GRAPH_EXECUTION_DOMAIN_DRAFT;
        case LLAMA_CONTEXT_TYPE_MTP:   return GGML_GRAPH_EXECUTION_DOMAIN_MTP;
        case LLAMA_CONTEXT_TYPE_DEFAULT: break;
    }
    return GGML_GRAPH_EXECUTION_DOMAIN_INVALID;
}

static llama_speculative_execution_policy speculative_execution_policy_for(
        llama_context_type context_type,
        const llama_batch & batch,
        uint32_t cache_slots,
        bool backend_supported) {
    llama_speculative_execution_policy result;
    result.domain = speculative_execution_domain(context_type);
    if (result.domain == GGML_GRAPH_EXECUTION_DOMAIN_INVALID) {
        return result;
    }
    result.flags = required_grouped_execution_flags(cache_slots, backend_supported);
    if (cache_slots == 0) {
        return result;
    }
    result.row_semantics = speculative_batch_row_semantics(batch);
    if (result.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INVALID) {
        result.fail_closed = result.flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_REQUIRED_GROUPED;
        return result;
    }
    result.preserve_intent = true;
    return result;
}

static bool batch_init_with_speculative_execution_policy(
        llama_batch_allocr & balloc,
        llama_context_type context_type,
        const llama_batch & batch,
        const llama_vocab & vocab,
        const llama_memory_i * memory,
        uint32_t n_embd,
        uint32_t n_seq_max,
        bool output_all,
        uint32_t cache_slots,
        bool backend_supported,
        llama_speculative_execution_policy * policy) {
    if (!balloc.init(batch, vocab, memory, n_embd, n_seq_max, output_all)) {
        return false;
    }
    if (policy != nullptr) {
        *policy = speculative_execution_policy_for(
            context_type, balloc.get_batch(), cache_slots, backend_supported);
    }
    return true;
}

static bool batch_init_with_speculative_execution_policy(
        llama_batch_allocr & balloc,
        llama_context_type context_type,
        const llama_batch_ext & batch,
        const llama_vocab & vocab,
        bool output_all,
        uint32_t cache_slots,
        bool backend_supported,
        llama_speculative_execution_policy * policy) {
    if (!balloc.init(batch, vocab, output_all)) {
        return false;
    }
    if (policy != nullptr) {
        *policy = speculative_execution_policy_for(
            context_type, balloc.get_batch(), cache_slots, backend_supported);
    }
    return true;
}

static bool target_verification_intent_valid(
        const llama_decode_execution_intent * intent,
        const llama_batch & batch,
        llama_graph_execution_intent & execution) {
    if (intent == nullptr || intent->magic != LLAMA_DECODE_EXECUTION_INTENT_MAGIC ||
            intent->abi_version != LLAMA_DECODE_EXECUTION_INTENT_VERSION ||
            intent->struct_size != sizeof(*intent) || intent->type != LLAMA_DECODE_EXECUTION_INTENT_TARGET_VERIFICATION ||
            intent->flags != LLAMA_DECODE_EXECUTION_INTENT_FLAG_NONE || intent->verification_span <= 1 ||
            batch.n_tokens <= 0 || batch.n_tokens % intent->verification_span != 0 ||
            batch.token == nullptr || batch.embd != nullptr || batch.pos == nullptr || batch.n_seq_id == nullptr ||
            batch.seq_id == nullptr || batch.logits == nullptr) {
        return false;
    }
    for (uint64_t value : intent->reserved) {
        if (value != 0) {
            return false;
        }
    }

    const uint32_t n_sequences = batch.n_tokens / intent->verification_span;
    for (uint32_t sequence = 0; sequence < n_sequences; ++sequence) {
        const uint32_t first = sequence * intent->verification_span;
        if (batch.n_seq_id[first] != 1 || batch.seq_id[first] == nullptr) {
            return false;
        }
        const llama_seq_id seq_id = batch.seq_id[first][0];
        if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ) {
            return false;
        }
        const llama_pos pos = batch.pos[first];
        for (uint32_t previous = 0; previous < sequence; ++previous) {
            const uint32_t previous_first = previous * intent->verification_span;
            if (batch.seq_id[previous_first][0] == seq_id) {
                return false;
            }
        }
        for (uint32_t offset = 0; offset < intent->verification_span; ++offset) {
            const uint32_t row = first + offset;
            if (batch.n_seq_id[row] != 1 || batch.seq_id[row] == nullptr ||
                    batch.seq_id[row][0] != seq_id || !verification_position_matches(pos, offset, batch.pos[row]) ||
                    batch.logits[row] == 0) {
                return false;
            }
        }
    }

    execution.domain = GGML_GRAPH_EXECUTION_DOMAIN_MAIN;
    execution.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE;
    execution.verification_span = intent->verification_span;
    return true;
}

static bool ubatch_has_target_verification_spans(const llama_ubatch & ubatch, uint32_t span) {
    if (span <= 1 || ubatch.n_tokens == 0 || ubatch.n_tokens % span != 0 || ubatch.token == nullptr ||
            ubatch.embd != nullptr || ubatch.pos == nullptr || ubatch.output == nullptr || ubatch.n_seq_id == nullptr ||
            ubatch.seq_id == nullptr || ubatch.seq_id_unq == nullptr) {
        return false;
    }
    const uint32_t n_sequences = ubatch.n_tokens / span;
    const bool equal_shape = ubatch.n_seq_tokens == span && ubatch.n_seqs == n_sequences;
    const bool simple_shape = ubatch.n_seq_tokens == 1 && ubatch.n_seqs == ubatch.n_tokens;
    if ((!equal_shape && !simple_shape) || ubatch.n_seqs_unq != n_sequences) {
        return false;
    }
    std::array<uint8_t, LLAMA_MAX_SEQ> listed = {};
    for (uint32_t sequence = 0; sequence < ubatch.n_seqs_unq; ++sequence) {
        const llama_seq_id seq_id = ubatch.seq_id_unq[sequence];
        if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ || listed[seq_id]) {
            return false;
        }
        listed[seq_id] = 1;
    }
    std::array<uint8_t, LLAMA_MAX_SEQ> seen = {};
    for (uint32_t sequence = 0; sequence < n_sequences; ++sequence) {
        const uint32_t first = sequence * span;
        if (ubatch.n_seq_id[first] != 1 || ubatch.seq_id[first] == nullptr) {
            return false;
        }
        const llama_seq_id seq_id = ubatch.seq_id[first][0];
        if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ || !listed[seq_id] || seen[seq_id]) {
            return false;
        }
        seen[seq_id] = 1;
        const llama_pos pos = ubatch.pos[first];
        for (uint32_t offset = 0; offset < span; ++offset) {
            const uint32_t row = first + offset;
            if (ubatch.n_seq_id[row] != 1 || ubatch.seq_id[row] == nullptr || ubatch.seq_id[row][0] != seq_id ||
                    !verification_position_matches(pos, offset, ubatch.pos[row]) || !ubatch.output[row]) {
                return false;
            }
        }
    }
    return true;
}

static bool ubatch_has_speculative_independent_rows(const llama_ubatch & ubatch) {
    return ubatch_has_independent_rows(ubatch) && ubatch.n_seqs == ubatch.n_tokens &&
        ubatch.n_seqs_unq == ubatch.n_tokens && ubatch.n_seq_tokens == 1;
}

static bool ubatch_has_sequential_spans(const llama_ubatch & ubatch) {
    if (ubatch.n_tokens == 0 || ubatch.pos == nullptr || ubatch.n_seq_id == nullptr || ubatch.seq_id == nullptr ||
            ubatch.seq_id_unq == nullptr || ubatch.n_seqs_unq == 0 || ubatch.n_seqs_unq > ubatch.n_tokens) {
        return false;
    }
    std::array<uint8_t, LLAMA_MAX_SEQ> listed = {};
    for (uint32_t sequence = 0; sequence < ubatch.n_seqs_unq; ++sequence) {
        const llama_seq_id seq_id = ubatch.seq_id_unq[sequence];
        if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ || listed[seq_id]) {
            return false;
        }
        listed[seq_id] = 1;
    }
    std::array<uint8_t, LLAMA_MAX_SEQ> seen = {};
    llama_seq_id current_seq = -1;
    llama_pos current_pos = 0;
    uint32_t n_spans = 0;
    for (uint32_t row = 0; row < ubatch.n_tokens; ++row) {
        if (ubatch.n_seq_id[row] != 1 || ubatch.seq_id[row] == nullptr) {
            return false;
        }
        const llama_seq_id seq_id = ubatch.seq_id[row][0];
        if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ || !listed[seq_id]) {
            return false;
        }
        if (seq_id != current_seq) {
            if (seen[seq_id]) {
                return false;
            }
            seen[seq_id] = 1;
            current_seq = seq_id;
            ++n_spans;
        } else if (!verification_position_matches(current_pos, 1, ubatch.pos[row])) {
            return false;
        }
        current_pos = ubatch.pos[row];
    }
    return n_spans == ubatch.n_seqs_unq;
}

static bool ubatch_matches_graph_execution_intent(
        llama_context_type context_type,
                  uint32_t cache_slots,
      const llama_ubatch & ubatch,
      const llama_graph_execution_intent & execution_intent) {
    if (execution_intent.domain == GGML_GRAPH_EXECUTION_DOMAIN_MAIN) {
        return context_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
            execution_intent.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE &&
            ubatch_has_target_verification_spans(ubatch, execution_intent.verification_span);
    }
    if (execution_intent.domain == GGML_GRAPH_EXECUTION_DOMAIN_DRAFT ||
            execution_intent.domain == GGML_GRAPH_EXECUTION_DOMAIN_MTP) {
        return execution_intent.domain == speculative_execution_domain(context_type) && cache_slots > 0 &&
            ((execution_intent.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT &&
                ubatch_has_speculative_independent_rows(ubatch)) ||
             (execution_intent.row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL &&
                ubatch_has_sequential_spans(ubatch)));
    }
    return false;
}

uint32_t llama_speculative_grouped_intent_test_access::classify_batch(const llama_batch & batch) {
    return speculative_batch_row_semantics(batch);
}

bool llama_speculative_grouped_intent_test_access::matches_ubatch(
        const llama_ubatch & ubatch,
        uint32_t row_semantics) {
    return matches_ubatch(LLAMA_CONTEXT_TYPE_MTP, ubatch, row_semantics);
}

bool llama_speculative_grouped_intent_test_access::matches_ubatch(
        llama_context_type context_type,
        const llama_ubatch & ubatch,
        uint32_t row_semantics) {
    llama_graph_execution_intent execution_intent;
    execution_intent.domain = speculative_execution_domain(context_type);
    execution_intent.row_semantics = row_semantics;
    return ubatch_matches_graph_execution_intent(context_type, 1, ubatch, execution_intent);
}

bool llama_speculative_grouped_intent_test_access::matches_target_verification_ubatch(
        const llama_ubatch & ubatch,
        uint32_t verification_span) {
    llama_graph_execution_intent execution_intent;
    execution_intent.domain = GGML_GRAPH_EXECUTION_DOMAIN_MAIN;
    execution_intent.row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE;
    execution_intent.verification_span = verification_span;
    return ubatch_matches_graph_execution_intent(
        LLAMA_CONTEXT_TYPE_DEFAULT, 1, ubatch, execution_intent);
}

bool llama_speculative_grouped_intent_test_access::backend_supported(ggml_backend_t backend) {
    return backend_supports_required_grouped_execution(backend);
}

bool llama_speculative_grouped_intent_test_access::hybrid_required(const char * mode, llama_context_type context_type) {
    return moe_hybrid_required_for_context(mode, context_type);
}

bool llama_speculative_grouped_intent_test_access::hybrid_execution_supported(
        bool required, uint32_t domain, uint32_t row_semantics) {
    return moe_hybrid_execution_supported(required, domain, row_semantics);
}

bool llama_speculative_grouped_intent_test_access::graph_supported(ggml_backend_sched_t sched, ggml_cgraph * gf) {
    return graph_supports_required_grouped_execution(sched, gf);
}

uint32_t llama_speculative_grouped_intent_test_access::flags(uint32_t cache_slots, bool backend_supported) {
    return required_grouped_execution_flags(cache_slots, backend_supported);
}

llama_speculative_execution_policy llama_speculative_grouped_intent_test_access::policy(
        const llama_batch & batch,
        uint32_t cache_slots,
        bool backend_supported) {
    return policy(LLAMA_CONTEXT_TYPE_MTP, batch, cache_slots, backend_supported);
}

llama_speculative_execution_policy llama_speculative_grouped_intent_test_access::policy(
        llama_context_type context_type,
        const llama_batch & batch,
        uint32_t cache_slots,
        bool backend_supported) {
    return speculative_execution_policy_for(context_type, batch, cache_slots, backend_supported);
}

bool llama_speculative_grouped_intent_test_access::policy_after_batch_init(
        const llama_batch & batch,
        const llama_vocab & vocab,
        uint32_t cache_slots,
        bool backend_supported,
        llama_speculative_execution_policy & policy) {
    return policy_after_batch_init(
        LLAMA_CONTEXT_TYPE_MTP, batch, vocab, cache_slots, backend_supported, policy);
}

bool llama_speculative_grouped_intent_test_access::policy_after_batch_init(
        llama_context_type context_type,
        const llama_batch & batch,
        const llama_vocab & vocab,
        uint32_t cache_slots,
        bool backend_supported,
        llama_speculative_execution_policy & policy,
        bool output_all) {
    llama_batch_allocr balloc(1);
    return batch_init_with_speculative_execution_policy(
        balloc, context_type, batch, vocab, nullptr, 1, 4, output_all, cache_slots, backend_supported, &policy);
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_AR,
    /*.name             =*/ "fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_CH,
    /*.name             =*/ "fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/ 16,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

using backend_flash_attn_causal_prefix_supported_t = bool (*)(ggml_backend_dev_t);

static bool llama_sched_supports_flash_attn_causal_prefix(ggml_backend_sched_t sched) {
    const int n_backends = ggml_backend_sched_get_n_backends(sched);
    for (int i = 0; i < n_backends; ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto fn = reg ? reinterpret_cast<backend_flash_attn_causal_prefix_supported_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_flash_attn_causal_prefix_supported")) : nullptr;
        if (!fn || !fn(dev)) {
            return false;
        }
    }

    return true;
}

llama_moe_candidate_snapshot::llama_moe_candidate_snapshot(
        const llama_model & model,
        const llama_adapter_loras & loras) {
    snapshot.magic = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_MAGIC;
    snapshot.abi_version = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_VERSION;
    snapshot.struct_size = sizeof(snapshot);
    snapshot.n_slots = std::max(model.moe_expert_cache_slots(), 0);
    const bool tensor_overrides = model.has_tensor_overrides();

    auto has_lora = [&](ggml_tensor * tensor) {
        for (const auto & lora : loras) {
            if (lora.second != 0.0f && lora.first != nullptr && lora.first->get_weight(tensor) != nullptr) {
                return true;
            }
        }
        return false;
    };

    const auto is_cached = is_moe_cached_tensor;

    std::unordered_set<const ggml_tensor *> seen;
    groups.reserve(std::min<size_t>(model.layers.size() * 2, GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS));
    tensors.reserve(std::min<size_t>(model.layers.size() * 12, GGML_BACKEND_MOE_CANDIDATE_MAX_TENSORS_V2));

    auto mark_typed_alias = [&](const ggml_tensor * tensor) {
        snapshot.flags |= GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_INCOMPLETE;
        for (const auto & record : tensors) {
            if (record.tensor == tensor && record.group_index < groups.size()) {
                groups[record.group_index].flags |= GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_INCOMPLETE;
            }
        }
    };

    auto append_group = [&](const llama_moe_source_group & source) {
        if (groups.size() == GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS) {
            snapshot.flags |= GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_INCOMPLETE;
            return;
        }

        const uint32_t group_index = groups.size();
        uint32_t group_flags = 0;
        if (source.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_INVALID || !source.route_present) {
            group_flags |= GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_INCOMPLETE;
        }
        groups.push_back({source.layout, source.domain, group_flags, 0});

        auto append_tensor = [&](ggml_tensor * tensor, uint32_t role, uint32_t status) {
            if (tensor == nullptr) {
                return;
            }
            if (tensors.size() == GGML_BACKEND_MOE_CANDIDATE_MAX_TENSORS_V2) {
                groups[group_index].flags |= GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_INCOMPLETE;
                snapshot.flags |= GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_INCOMPLETE;
                return;
            }
            if (!seen.insert(tensor).second) {
                groups[group_index].flags |= GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_INCOMPLETE;
                mark_typed_alias(tensor);
                return;
            }
            const bool cached = is_cached(tensor);
            const bool overridden = tensor_overrides && !cached;
            uint32_t flags = overridden ? GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_TENSOR_OVERRIDES : 0;
            if (overridden) {
                groups[group_index].flags |= GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_TENSOR_OVERRIDES;
            }
            if (cached) {
                flags |= GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER;
            }
            if (has_lora(tensor)) {
                flags |= GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_ACTIVE_LORA;
                if (status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                    groups[group_index].flags |= GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_ACTIVE_LORA;
                }
            }
            tensors.push_back({tensor, group_index, role, status, flags, 0});
        };

        for (const auto & bank : source.banks) {
            append_tensor(bank.tensor, bank.role, bank.status);
        }
    };

    for (const auto & source : model.moe_sources()) {
        append_group(source);
    }

    auto append_excluded = [&](ggml_tensor * tensor, uint32_t status) {
        if (tensor == nullptr) {
            return;
        }
        if (!seen.insert(tensor).second) {
            mark_typed_alias(tensor);
            return;
        }
        if (tensors.size() == GGML_BACKEND_MOE_CANDIDATE_MAX_TENSORS_V2) {
            snapshot.flags |= GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_INCOMPLETE;
            return;
        }
        const bool cached = is_cached(tensor);
        uint32_t flags = tensor_overrides && !cached ? GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_TENSOR_OVERRIDES : 0;
        if (cached) {
            flags |= GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER;
        }
        if (has_lora(tensor)) {
            flags |= GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_ACTIVE_LORA;
        }
        tensors.push_back({tensor, UINT32_MAX, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID, status, flags, 0});
    };

    for (const auto & layer : model.layers) {
        append_excluded(layer.ffn_gate_shexp, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_SHARED);
        append_excluded(layer.ffn_up_shexp, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_SHARED);
        append_excluded(layer.ffn_down_shexp, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_SHARED);
        append_excluded(layer.ffn_gate, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_DENSE);
        append_excluded(layer.ffn_up, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_DENSE);
        append_excluded(layer.ffn_down, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_EXCLUDED_DENSE);
    }

    for (const auto & named_tensor : model.tensors_by_name) {
        const ggml_tensor * tensor = named_tensor.second;
        if (!is_cached(tensor) || !seen.insert(tensor).second) {
            continue;
        }
        if (tensors.size() == GGML_BACKEND_MOE_CANDIDATE_MAX_TENSORS_V2) {
            snapshot.flags |= GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_FLAG_INCOMPLETE;
            break;
        }
        uint32_t flags = GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_CACHED_BUFFER;
        if (has_lora(const_cast<ggml_tensor *>(tensor))) {
            flags |= GGML_BACKEND_MOE_CANDIDATE_TENSOR_V2_FLAG_ACTIVE_LORA;
        }
        tensors.push_back({tensor, UINT32_MAX, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID,
            GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_UNCLASSIFIED, flags, 0});
    }

    snapshot.groups = groups.empty() ? nullptr : groups.data();
    snapshot.n_groups = static_cast<uint32_t>(groups.size());
    snapshot.tensors = tensors.empty() ? nullptr : tensors.data();
    snapshot.n_tensors = static_cast<uint32_t>(tensors.size());
}

const ggml_backend_moe_candidate_snapshot_v2 & llama_moe_candidate_snapshot::get() const {
    return snapshot;
}

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params,
              const char * profile_path,
              const char * profile_adaptation,
              const char * cache_allocation) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    // MTP uses the embd input for the hidden state
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd(),
                llm_arch_supports_mixed_batch(model.arch) && params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT)) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    graph_execution_owner_namespace = graph_execution_next_owner_namespace();
    const char * moe_hybrid = getenv("GGML_MOE_HYBRID_METADATA");
    moe_hybrid_metadata = moe_hybrid != nullptr && atoi(moe_hybrid) != 0;
    const char * hybrid_mode = getenv("GGML_MOE_HYBRID");
    moe_hybrid_required = moe_hybrid_required_for_context(hybrid_mode, params.ctx_type);
    moe_source_graph_capacity = params.moe_source_graph_capacity;
    moe_hybrid_metadata |= moe_hybrid_required;
    if (moe_hybrid_required) {
        const char * executor = getenv("GGML_MOE_HYBRID_EXECUTOR");
        if (!executor) { executor = "source"; }
        const bool source_executor = strcmp(executor, "source") == 0;
        if (strcmp(executor, "eager") != 0 && strcmp(executor, "fidelity") != 0 && !source_executor) {
            throw std::runtime_error("GGML_MOE_HYBRID_EXECUTOR must be eager, source or fidelity");
        }
        moe_hybrid_executor = strcmp(executor, "fidelity") == 0 || source_executor ?
            GGML_BACKEND_MOE_HYBRID_EXECUTOR_V1_FIDELITY : GGML_BACKEND_MOE_HYBRID_EXECUTOR_V1_EAGER;
        if (source_executor && !ggml_moe_fidelity_selection().valid) {
            throw std::runtime_error("source hybrid execution requires a finite GPU miss fraction in [0,1] and compatible pipeline settings");
        }
        if (moe_hybrid_executor == GGML_BACKEND_MOE_HYBRID_EXECUTOR_V1_FIDELITY && !source_executor) {
            const char * live_experiment = getenv("GGML_MOE_FIDELITY_LIVE_EXPERIMENT");
            if (live_experiment == nullptr || strcmp(live_experiment, "1") != 0) {
                throw std::runtime_error("fidelity hybrid execution requires GGML_MOE_FIDELITY_LIVE_EXPERIMENT=1");
            }
        }
        if (const char * prefill = getenv("GGML_MOE_SOURCE_CPU_PREFILL")) {
            if (strcmp(prefill, "0") && strcmp(prefill, "1")) { throw std::runtime_error("GGML_MOE_SOURCE_CPU_PREFILL must be 0 or 1"); }
            if (!strcmp(prefill, "1") && !source_core_enabled()) { throw std::runtime_error("CPU-assisted prefill requires source hybrid execution"); }
        }
        moe_hybrid_allow_runtime_allocations = source_core_enabled();
        if (const char * allocations = getenv("GGML_MOE_HYBRID_ALLOW_RUNTIME_ALLOCATIONS")) {
            if (strcmp(allocations, "0") && strcmp(allocations, "1")) {
                throw std::runtime_error("GGML_MOE_HYBRID_ALLOW_RUNTIME_ALLOCATIONS must be 0 or 1");
            }
            moe_hybrid_allow_runtime_allocations = !strcmp(allocations, "1");
        }
        if (const char * batch = getenv("GGML_MOE_HYBRID_RESIDENT_BATCH")) {
            if (strcmp(batch, "off") != 0 && strcmp(batch, "on") != 0) {
                throw std::runtime_error("GGML_MOE_HYBRID_RESIDENT_BATCH must be off or on");
            }
            moe_hybrid_resident_batch = strcmp(batch, "on") == 0;
        }
        if (const char * admission = getenv("GGML_MOE_HYBRID_ADMISSION")) {
            if (strcmp(admission, "off") != 0 && strcmp(admission, "demand") != 0) {
                throw std::runtime_error("GGML_MOE_HYBRID_ADMISSION must be off or demand");
            }
            moe_hybrid_demand_admission = strcmp(admission, "demand") == 0;
        }
        if (const char * quota = getenv("GGML_MOE_HYBRID_GPU_MISSES")) {
            char * end = nullptr;
            const unsigned long long value = strtoull(quota, &end, 10);
            if (quota[0] < '0' || quota[0] > '9' || *end != '\0' || value > UINT32_MAX) {
                throw std::runtime_error("GGML_MOE_HYBRID_GPU_MISSES must be an unsigned 32-bit integer");
            }
            moe_hybrid_gpu_misses = value;
        }
        if (const char * quota = getenv("GGML_MOE_HYBRID_ADMISSION_MISSES")) {
            char * end = nullptr;
            const unsigned long long value = strtoull(quota, &end, 10);
            if (quota[0] < '0' || quota[0] > '9' || *end != '\0' || value > UINT32_MAX) {
                throw std::runtime_error("GGML_MOE_HYBRID_ADMISSION_MISSES must be an unsigned 32-bit integer");
            }
            moe_hybrid_admission_misses = value;
        }
        if (moe_hybrid_admission_misses == UINT32_MAX) {
            moe_hybrid_admission_misses = moe_hybrid_gpu_misses;
        }
        if (moe_hybrid_admission_misses > moe_hybrid_gpu_misses) {
            throw std::runtime_error("GGML_MOE_HYBRID_ADMISSION_MISSES must not exceed GGML_MOE_HYBRID_GPU_MISSES");
        }
    }

    if (!cache_allocation || (strcmp(cache_allocation, "auto") && strcmp(cache_allocation, "uniform"))) {
        throw std::runtime_error("MoE cache allocation must be auto or uniform");
    }
    const char * path = profile_path;
    const char * adapt = profile_adaptation;
    if (!profile_path && params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT) {
        path = getenv("GGML_MOE_EXPERT_PROFILE");
        adapt = getenv("GGML_MOE_HYBRID_PROFILE_ADAPT");
    }
    if (adapt && (!profile_path || strcmp(adapt, "off"))) {
        if ((strcmp(adapt, "occurrence") && strcmp(adapt, "occurrence-sync")) || !source_core_enabled() || !path) {
            throw std::runtime_error("MoE profile adaptation requires occurrence or occurrence-sync, a profile and generic source execution");
        }
        moe_hybrid_profile_adapt = !strcmp(adapt, "occurrence") ? 2 : 1;
    }
    if (path) {
        if (!*path || (moe_hybrid_required && !source_core_enabled()) || model.moe_expert_cache_slots() <= 0) {
            throw std::runtime_error("GGML_MOE_EXPERT_PROFILE requires a positive cache capacity and a supported grouped executor");
        }
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        const auto extent = file ? file.tellg() : std::streampos(-1);
        if (extent < 24 || extent > 256 * 1024 * 1024) { throw std::runtime_error("MoE profile file is unavailable or too large"); }
        std::vector<uint8_t> bytes(static_cast<size_t>(extent));
        file.seekg(0);
        if (!file.read(reinterpret_cast<char *>(bytes.data()), bytes.size())) { throw std::runtime_error("MoE profile read failed"); }
        if (!memcmp(bytes.data(), "GGUF", 4)) {
            moe_profile_statistics = llama_moe_profile_statistics_parse(bytes.data(), bytes.size(), model.moe_sources());
            for (const auto & source : moe_profile_statistics.sources) {
                moe_statistics.push_back({source.tensor, source.counts.data(), source.observations, uint32_t(source.counts.size()), source.domain});
                if (!source.scores.empty()) { moe_statistics_scores.push_back(source.scores.data()); }
            }
            LLAMA_LOG_INFO("moe-profile: file=%s sources=%zu identity=source-name/domain/type/shape policy=canonical byte-weighted statistics use runtime capacities\n", path, moe_statistics.size());
        } else {
            std::vector<uint32_t> counts(model.hparams.n_layer(), 0);
            std::vector<const ggml_tensor *> down(counts.size(), nullptr);
            for (const auto & source : model.moe_sources()) {
                if (source.layer >= 0 && size_t(source.layer) >= counts.size()) { continue; }
                if (source.domain != GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY || !source.route_present ||
                        source.layer < 0 || size_t(source.layer) >= counts.size() || counts[source.layer]) {
                    throw std::runtime_error("STRP profile cannot bind ambiguous routing layer identities");
                }
                for (const auto & bank : source.banks) {
                    if (bank.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT) { down[source.layer] = bank.tensor; }
                }
                if (!down[source.layer] || down[source.layer]->ne[2] <= 0 || down[source.layer]->ne[2] > 65536) {
                    throw std::runtime_error("STRP profile requires a valid routing expert geometry");
                }
                counts[source.layer] = down[source.layer]->ne[2];
            }
            moe_profile_ranks = llama_moe_profile_parse(bytes.data(), bytes.size(), counts);
            for (size_t layer = 0; layer < counts.size(); ++layer) {
                if (!counts[layer]) { continue; }
                if (moe_profile_ranks[layer].size() < std::min<uint32_t>(std::max(model.moe_expert_cache_slots(model.dev_layer(layer)), 0), counts[layer])) {
                    throw std::runtime_error("STRP profile has insufficient ranked experts for the requested layer capacity");
                }
                moe_profiles.push_back({down[layer], moe_profile_ranks[layer].data(), uint32_t(moe_profile_ranks[layer].size())});
            }
            LLAMA_LOG_INFO("moe-profile: file=%s groups=%zu identity=geometry-only policy=static ranked IDs use runtime capacities\n", path, moe_profiles.size());
        }
        if (!strcmp(cache_allocation, "auto") && source_core_enabled()) {
            plan_moe_profile_capacities(params.ctx_type, bytes.data(), bytes.size());
        }
    }

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.kv_cpu_pinned           = params.kv_cpu_pinned;
    cparams.recurrent_state_offload = params.recurrent_state_offload;
    cparams.offload_attn_compute    = params.offload_kqv || (params.op_offload && params.kv_cpu_pinned);
    cparams.kv_gpu_layers           = params.kv_gpu_layers;
    cparams.phase_aware_workspace   = params.phase_aware_workspace;
    cparams.live_context_workspace  = params.live_context_workspace;
    cparams.decode_boundary_overlap = params.decode_boundary_overlap;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    embd_layer_inp.resize(hparams.n_layer() + 1);

    cparams.ctx_type          = params.ctx_type;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.pooling_type      = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        cparams.rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn                         = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.flash_attn_causal_prefix_supported = false;
    cparams.auto_fa                            = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = false;

    cparams.fused_lid = true;
    cparams.auto_flid = false;

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);
    sched_decode_outputs = llama_model_has_encoder(&model) ? cparams.n_batch :
            params.n_outputs_max != 0 ? params.n_outputs_max :
            cparams.phase_aware_workspace ? cparams.n_seq_max : cparams.n_outputs_max;

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    cparams.op_offload     = params.op_offload;
    cparams.kv_unified     = params.kv_unified;
    cparams.moe_cache_size = params.moe_cache_size;
    if (cparams.moe_cache_size > 0 && model.moe_expert_cache_slots() > 0) {
        throw std::runtime_error("--moe-cache-mib cannot be combined with --moe-expert-cache-size");
    }

    // initialized later
    cparams.pipeline_parallel = false;
    cparams.training = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
    LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                if (cparams.decode_boundary_overlap) {
                    auto enable = reinterpret_cast<void (*)(ggml_backend_t, bool)>(
                        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_set_decode_boundary_overlap"));
                    if (enable) {
                        enable(backend.get(), true);
                        LLAMA_LOG_INFO("%s: decode boundary overlap enabled for %s\n", __func__, ggml_backend_name(backend.get()));
                    }
                }
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }

                auto moe_candidate_replace_fn = (ggml_backend_moe_candidate_replace_v2_t) ggml_backend_reg_get_proc_address(
                        reg, GGML_BACKEND_MOE_CANDIDATE_REPLACE_V2_PROC_NAME);
                if (moe_candidate_replace_fn) {
                    auto set_max_rows = (ggml_backend_moe_early_router_set_max_rows_t) ggml_backend_reg_get_proc_address(
                            reg, GGML_BACKEND_MOE_EARLY_ROUTER_SET_MAX_ROWS_PROC_NAME);
                    if (set_max_rows) {
                        set_max_rows(backend.get(), model.moe_early_router_max_rows());
                    }
                    moe_candidate_replace_fns.emplace_back(backend.get(), moe_candidate_replace_fn);
                }
            }
        }

        refresh_moe_layer_owners();
        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
        };

        memory.reset(model.create_memory(params_mem, cparams));

        if (cparams.live_context_workspace && hparams.no_alloc) {
            cparams.live_context_workspace = false;
        } else if (cparams.live_context_workspace && (!memory || memory->get_attn_reserve_capacity() == 0)) {
            LLAMA_LOG_WARN("%s: live-context workspace sizing unsupported; using full-context reserve\n", __func__);
            cparams.live_context_workspace = false;
        }

        if (!cparams.offload_kqv && cparams.kv_gpu_layers > 0) {
            if (memory && memory->get_supports_partial_kv()) {
                cparams.offload_attn_compute = cparams.offload_attn_compute || cparams.op_offload;
            } else {
                LLAMA_LOG_WARN("%s: partial GPU KV residency is not supported for this memory layout; ignoring kv_gpu_layers\n", __func__);
            }
        }
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides() &&
            cparams.moe_cache_size == 0; // not supported by the MoE cache

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        if (cparams.moe_cache_size > 0) {
            moe_cache = std::make_unique<llama_moe_cache>(model, backend_ptrs, backend_buft, cparams.moe_cache_size);
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    if (source_core_enabled() && moe_statistics.empty() && moe_profiles.empty() &&
            params.ctx_other && params.ctx_other->source_core_enabled() &&
            &params.ctx_other->model == &model) {
        llama_moe_profile_statistics statistics;
        uint32_t adaptation = 0;
        {
            auto & parent = *params.ctx_other;
            std::unique_lock<std::timed_mutex> publication_lock(parent.moe_source_publication_mutex, std::try_to_lock);
            if (!publication_lock.owns_lock()) { throw std::runtime_error("cannot snapshot an active source profile publication"); }
            std::lock_guard<std::mutex> caller_lock(parent.moe_source_mutex);
            if (parent.moe_source_callers || parent.moe_source_closed.load() || parent.moe_source_poisoned.load() || parent.moe_profile_failed) {
                throw std::runtime_error("cannot inherit source statistics from an unavailable context");
            }
            statistics = parent.moe_profile_statistics;
            adaptation = parent.moe_hybrid_profile_adapt;
        }
        if (!statistics.sources.empty()) {
            std::vector<ggml_backend_moe_source_statistics_v1> views;
            std::vector<const double *> scores;
            views.reserve(statistics.sources.size());
            for (const auto & source : statistics.sources) {
                views.push_back({source.tensor, source.counts.data(), source.observations, uint32_t(source.counts.size()), source.domain});
                if (!source.scores.empty()) { scores.push_back(source.scores.data()); }
            }
            if (!initialize_moe_statistics(views, scores.empty() ? nullptr : scores.data())) { throw std::runtime_error("cannot install inherited source statistics"); }
            for (size_t i = 0; i < statistics.sources.size(); ++i) {
                auto & input = statistics.sources[i];
                auto & output = moe_profile_statistics.sources[i];
                output.heat = std::move(input.heat); output.usage = std::move(input.usage);
                output.prior = std::move(input.prior); output.windows = input.windows;
            }
            moe_profile_statistics.provenance = std::move(statistics.provenance);
            moe_hybrid_profile_adapt = adaptation;
            LLAMA_LOG_INFO("moe-profile: inherited sources=%zu identity=same-model-object adaptation=%u storage=context-owned\n",
                moe_statistics.size(), moe_hybrid_profile_adapt);
        }
    }

    if (!restore_moe_learning()) { throw std::runtime_error("cannot restore learned source profile before execution"); }
    if (source_core_enabled() && !initialize_moe_placement(moe_profiles, moe_statistics,
            moe_statistics_scores.empty() ? nullptr : moe_statistics_scores.data(), true)) {
        throw std::runtime_error("cannot initialize source profile residency before execution");
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    GGML_ASSERT(close_source_core_checked() == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();

    // when training, ggml_opt allocates extra buffers through the scheduler, so the sizes no longer match the expectation
    if (!model.hparams.no_alloc && !opt_ctx) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (cparams.phase_aware_workspace || cparams.live_context_workspace) {
                LLAMA_LOG_DEBUG("%s: %10s resizable compute backing size is %8.4f MiB (last observed %8.4f MiB)\n",
                        __func__, ggml_backend_buft_name(buft),
                        size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);

    if (sched_buffer_owner != nullptr && sched_buffer_owner->sched_buffer_borrower == this) {
        sched_buffer_owner->sched_buffer_borrower = nullptr;
    }
    if (sched_buffer_borrower != nullptr && sched_buffer_borrower->sched_buffer_owner == this) {
        sched_buffer_borrower->sched_buffer_owner = nullptr;
        sched_buffer_borrower->sched_buffers_shared = false;
    }
}

bool llama_context::set_mtp_draft_vocab(const char * path) {
    if (cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP || mtp_draft_vocab || mtp_draft_vocab_locked || workspace_in_flight) {
        LLAMA_LOG_ERROR("%s: configure an MTP context before its first decode\n", __func__);
        return false;
    }
    try {
        mtp_draft_vocab = std::make_unique<llama_draft_vocab>(model, path);
        return true;
    } catch (const std::exception & error) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, error.what());
        return false;
    }
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;

    const auto supports_sparse_gdn = [](ggml_backend_dev_t device, const ggml_tensor * op) {
        if (device == nullptr || op == nullptr || op->op != GGML_OP_GATED_DELTA_NET || op->src[2] == nullptr) {
            return false;
        }

        int32_t n_snapshots;
        memcpy(&n_snapshots, op->op_params, sizeof(n_snapshots));
        const int64_t n_tokens = op->src[2]->ne[2];
        if (n_snapshots <= 1 || n_tokens <= 0) {
            return false;
        }

        ggml_tensor probe = *op;
        ggml_gated_delta_net_set_snapshots(
                &probe, (int32_t) std::min<int64_t>(n_tokens, n_snapshots - 1), -1, true);
        if (!ggml_backend_dev_supports_op(device, &probe)) {
            return false;
        }

        ggml_gated_delta_net_set_snapshots(&probe, 0, 0, false);
        return ggml_backend_dev_supports_op(device, &probe);
    };

    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled, bool * sparse_snapshot_backend_supported = nullptr) {
        if (!enabled) {
            if (sparse_snapshot_backend_supported) {
                *sparse_snapshot_backend_supported = false;
            }
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        if (sparse_snapshot_backend_supported) {
            bool found = false;
            bool supported = true;
            for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
                if (node.op != probe.op) {
                    continue;
                }

                found = true;
                ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
                ggml_backend_dev_t device = backend ? ggml_backend_get_device(backend) : nullptr;
                supported = supported && supports_sparse_gdn(device, node.tensor);
            }
            *sparse_snapshot_backend_supported = found && supported;
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        bool sparse_snapshot_ar_supported = false;
        bool sparse_snapshot_ch_supported = false;
        resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar, &sparse_snapshot_ar_supported);
        resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch, &sparse_snapshot_ch_supported);
        recurrent_sparse_snapshot_ops_supported = sparse_snapshot_ar_supported && sparse_snapshot_ch_supported;
        cparams.auto_fgdn = false;
    } else if (cparams.n_rs_seq > 0 && model.graph_supports_recurrent_sparse_snapshots()) {
        bool sparse_snapshot_ar_supported = false;
        bool sparse_snapshot_ch_supported = false;
        bool fused_gdn_ar = cparams.fused_gdn_ar;
        bool fused_gdn_ch = cparams.fused_gdn_ch;
        resolve(llm_fused_op_gdn_ar_probe, fused_gdn_ar, &sparse_snapshot_ar_supported);
        resolve(llm_fused_op_gdn_ch_probe, fused_gdn_ch, &sparse_snapshot_ch_supported);
        recurrent_sparse_snapshot_ops_supported = sparse_snapshot_ar_supported && sparse_snapshot_ch_supported;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe,  cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }
}

static int llama_graph_n_input_tensors(ggml_cgraph * gf) {
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            users[node].push_back(node);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = node->src[j];
            if (!src) {
                break;
            }
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                users[src].push_back(node);
            }
        }
    }

    for (const auto & [tensor, nodes] : users) {
        GGML_ASSERT(tensor->op == GGML_OP_NONE);
        for (const ggml_tensor * node : nodes) {
            LLAMA_LOG_DEBUG("%s: input tensor '%32s' [%s, ne = { %5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64 " }] is used by node '%s' (%s)\n",
                    __func__, tensor->name, ggml_type_name(tensor->type),
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    node->name, ggml_op_name(node->op));
        }
    }

    return (int) users.size();
}

static uint32_t llama_workspace_kv_growth_bound(uint32_t required, uint32_t capacity) {
    GGML_ASSERT(capacity > 0);
    required = std::max(1u, std::min(required, capacity));

    return llama_memory_graph_extent(required, capacity, 256u);
}

llama_context::sched_reserve_plan llama_context::make_sched_reserve_plan(
        uint32_t n_tokens_req,
        uint32_t n_kv_req) const {
    sched_reserve_plan plan;
    plan.n_tokens_max = std::min(cparams.n_ctx, cparams.n_ubatch);
    plan.n_tokens_decode = std::min(plan.n_tokens_max,
            std::max(cparams.n_seq_max, sched_decode_outputs));
    plan.n_tokens = plan.n_tokens_max;

    if (cparams.live_context_workspace) {
        GGML_ASSERT(memory);
        plan.n_kv_capacity = memory->get_attn_reserve_capacity();
        GGML_ASSERT(plan.n_kv_capacity > 0);
        plan.live_kv = true;
    }

    if (plan.live_kv) {
        uint32_t required = n_kv_req;
        if (required == 0) {
            required = sched_reserved_kv;
        }
        if (required == 0) {
            required = std::min(256u, plan.n_kv_capacity);
        }
        plan.n_kv = llama_workspace_kv_growth_bound(required, plan.n_kv_capacity);

        if (sched_reserved_kv > plan.n_kv && plan.n_kv >= sched_reserved_kv / 2) {
            plan.n_kv = sched_reserved_kv;
        }
    }

    if (cparams.phase_aware_workspace && !model.hparams.no_alloc) {
        plan.n_tokens = n_tokens_req > plan.n_tokens_decode ? plan.n_tokens_max : plan.n_tokens_decode;
        // Keep the prompt reservation for later prompt chunks that fit it.
        if (sched_reserved_tokens > plan.n_tokens_decode &&
                n_tokens_req > plan.n_tokens_decode && n_tokens_req <= sched_reserved_tokens) {
            plan.n_tokens = sched_reserved_tokens;
        }
    }

    return plan;
}

llama_context * llama_context::shared_workspace_peer() const {
    return sched_buffer_owner != nullptr ? sched_buffer_owner : sched_buffer_borrower;
}

void llama_context::reset_sched_workspace() {
    GGML_ASSERT(reset_source_core_checked() == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
    sched.reset();
    cparams.flash_attn_causal_prefix_supported = false;
    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_reserve.reset();
    gf_res_prev_active = nullptr;
    sched_buffer_generation = 0;
    sched_shrink_generation = 0;
    sched_source_retirement_epoch = 0;
    sched_buffers_shared = false;
    workspace_in_flight = false;
    sched_reserved_tokens = 0;
    sched_reserved_kv = 0;
    request_sched_reserve();
}

void llama_context::acquire_shared_workspace() {
    llama_context * peer = shared_workspace_peer();
    if (peer != nullptr && peer->workspace_in_flight) {
        peer->synchronize();
    }
}

void llama_context::prepare_sched_reserve(const sched_reserve_plan & plan) {
    if (!sched || (!cparams.phase_aware_workspace && !plan.live_kv) || model.hparams.no_alloc) {
        return;
    }

    if (sched_reserved_tokens > plan.n_tokens ||
            (plan.live_kv && sched_reserved_kv > plan.n_kv)) {
        ggml_backend_sched_request_buffer_shrink(sched.get());
    }

    uint64_t generation = 0;
    uint64_t shrink_generation = 0;
    ggml_backend_sched_get_buffer_state(sched.get(), &generation, &shrink_generation);

    // A backing generation change invalidates cached graph addresses.
    if (generation != sched_buffer_generation) {
        GGML_ASSERT(reset_source_core_checked(!source_graph_cache_enabled()) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
        synchronize();
        ggml_backend_sched_reset(sched.get());
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        gf_res_prev_active = nullptr;
        sched_buffer_generation = generation;
        request_sched_reserve();
    }

    if (shrink_generation != sched_shrink_generation) {
        sched_shrink_generation = shrink_generation;
        request_sched_reserve();
    }
}

void llama_context::request_sched_reserve(bool sampler_only) {
    if (!sampler_only) { sched_sampler_reserve_only = false; }
    else if (!sched_need_reserve) { sched_sampler_reserve_only = true; }
    sched_need_reserve = true;
}

void llama_context::sched_reserve(uint32_t n_tokens_req, uint32_t n_kv_req) {
    acquire_shared_workspace();

    const bool sched_resizable_requested = (cparams.phase_aware_workspace || cparams.live_context_workspace) &&
            !model.hparams.no_alloc;
    if (!sched_need_reserve && !sched_resizable_requested) {
        return;
    }

    const auto plan = make_sched_reserve_plan(n_tokens_req, n_kv_req);
    const bool sched_resizable = (cparams.phase_aware_workspace || plan.live_kv) && !model.hparams.no_alloc;
    if (!sched_need_reserve && !sched_resizable) {
        return;
    }

    prepare_sched_reserve(plan);

    if (!sched_need_reserve && sched_reserved_tokens == plan.n_tokens &&
            (!plan.live_kv || sched_reserved_kv == plan.n_kv)) {
        return;
    }

    sched_need_reserve = false;
    sched_sampler_reserve_only = false;

    const uint32_t n_tokens_max = plan.n_tokens_max;
    const uint32_t n_tokens_tg  = plan.n_tokens_decode;
    const uint32_t n_tokens     = plan.n_tokens;
    const uint32_t n_kv_capacity = plan.n_kv_capacity;
    const uint32_t n_kv          = plan.n_kv;
    const bool live_kv            = plan.live_kv;

    if (live_kv) {
        LLAMA_LOG_INFO("%s: reserving %s workspace (tokens = %u, previous = %u, kv = %u, previous = %u) ...\n",
                __func__, n_tokens == n_tokens_tg ? "decode" : "prefill",
                n_tokens, sched_reserved_tokens, n_kv, sched_reserved_kv);
    } else if (sched_resizable) {
        LLAMA_LOG_INFO("%s: reserving %s workspace (tokens = %u, previous = %u) ...\n",
                __func__, n_tokens == n_tokens_tg ? "decode" : "prefill",
                n_tokens, sched_reserved_tokens);
    } else {
        LLAMA_LOG_INFO("%s: reserving ...\n", __func__);
    }

    GGML_ASSERT(reset_source_core_checked() == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
    synchronize();
    refresh_moe_candidates();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;

    const size_t max_nodes = this->graph_max_nodes(n_tokens_max);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    if (sched_resizable) {
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        if (!gf_res_reserve) {
            gf_res_reserve.reset(new llm_graph_result(max_nodes));
        } else {
            gf_res_reserve->reset();
        }
    } else {
        for (auto & res : gf_res_prev) {
            res.reset();
        }
        gf_res_reserve.reset(new llm_graph_result(max_nodes));
    }
    gf_res_prev_active = nullptr;

    auto create_sched = [&](bool pipeline_parallel) {
        sched.reset(ggml_backend_sched_new(
                backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                max_nodes, pipeline_parallel, cparams.op_offload));
        ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);
        cparams.flash_attn_causal_prefix_supported = llama_sched_supports_flash_attn_causal_prefix(sched.get());
        if (sched_resizable) {
            sched_buffers_shared = sched_buffer_owner != nullptr && sched_buffer_owner->get_sched() != nullptr &&
                    ggml_backend_sched_set_resizable(sched.get(), sched_buffer_owner->get_sched());
            if (!sched_buffers_shared) {
                GGML_ASSERT(ggml_backend_sched_set_resizable(sched.get(), nullptr));
            }
            ggml_backend_sched_get_buffer_state(
                    sched.get(), &sched_buffer_generation, &sched_shrink_generation);
            LLAMA_LOG_INFO("%s: %s %s compute backing\n", __func__,
                    sched_buffers_shared ? "borrowing target" : "created",
                    cparams.phase_aware_workspace ? "phase-aware" : "live-context");
        }
    };

    if (!sched_resizable || !sched) {
        create_sched(cparams.pipeline_parallel);
    }

    llama_memory_context_ptr mctx;
    if (memory) {
        if (live_kv) {
            LLAMA_LOG_DEBUG("%s: reserving bounded memory module (n_kv = %u, capacity = %u)\n",
                    __func__, n_kv, n_kv_capacity);
            mctx = memory->init_reserve(n_kv);
        } else {
            LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
            mctx = memory->init_full();
        }
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    // reserve worst-case graph
    int n_splits_pp        = -1;
    int n_nodes_pp         = -1;
    int n_inputs_pp        = -1;
    int n_input_tensors_pp = -1;

    int n_splits_tg        = -1;
    int n_nodes_tg         = -1;
    int n_inputs_tg        = -1;
    int n_input_tensors_tg = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            const bool can_recreate = !sched_resizable ||
                    (sched_buffer_owner == nullptr && sched_buffer_borrower == nullptr);
            if (cparams.pipeline_parallel && can_recreate) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                create_sched(false);
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp         = ggml_graph_n_nodes(gf);
        n_inputs_pp        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_pp = this->n_input_tensors;
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    if (cparams.training) {
        // no tg graph for training
        n_splits_tg = n_splits_pp;
        n_nodes_tg  = n_nodes_pp;
    } else {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc || sched_resizable);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg         = ggml_graph_n_nodes(gf);
        n_inputs_tg        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_tg = this->n_input_tensors;
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: the worst case graph is not always reached for `n_seqs > 1`
        //       need to implement a more robust mechanism that tries a few different inputs and analyzes the results
        ggml_cgraph * gf = nullptr;
        switch (model.arch) {
            case LLM_ARCH_KIMI_LINEAR:
            case LLM_ARCH_MINIMAX_01:
                // [TAG_RESERVE_DIAG_DECAY]
                // the `inp_diag_decay` tensor size scales with `n_seq_tokens^2` which
                // makes `n_seqs == 1` use more memory for the compute graph compared to `n_seqs > 1`
                gf = graph_reserve(n_tokens, 1,      n_outputs_pp, mctx.get(),
                        model.hparams.no_alloc || sched_resizable);
                break;
            default:
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                        model.hparams.no_alloc || sched_resizable);
        };
        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    {
        const bool diff = n_nodes_pp != n_nodes_tg || n_splits_pp != n_splits_tg ||
                          n_inputs_pp != n_inputs_tg || n_input_tensors_pp != n_input_tensors_tg;

        const auto val = [diff](int v_pp, int v_tg) -> std::string {
            return diff ? format("%d / %d", v_pp, v_tg) : format("%d", v_pp);
        };

        LLAMA_LOG_INFO("%s: graph%s: nodes = %s, splits = %s, input objects = %s, input tensors = %s\n",
                __func__,
                diff ? format(" (pp bs=%d, tg bs=%d)", n_tokens, n_seqs).c_str() : "",
                val(n_nodes_pp, n_nodes_tg).c_str(),
                val(n_splits_pp, n_splits_tg).c_str(),
                val(n_inputs_pp, n_inputs_tg).c_str(),
                val(n_input_tensors_pp, n_input_tensors_tg).c_str());
    }

    const int64_t t_end_us = ggml_time_us();

    if (sched_resizable) {
        ggml_backend_sched_get_buffer_state(
                sched.get(), &sched_buffer_generation, &sched_shrink_generation);
    }
    sched_reserved_tokens = n_tokens;
    sched_reserved_kv = live_kv ? n_kv : 0;

    if (live_kv) {
        LLAMA_LOG_INFO("%s: %s workspace reserve took %.2f ms (kv = %u), sched copies = %d\n",
                __func__, n_tokens == n_tokens_tg ? "decode" : "prefill",
                (t_end_us - t_start_us)/1000.0, n_kv,
                ggml_backend_sched_get_n_copies(sched.get()));
    } else if (sched_resizable) {
        LLAMA_LOG_INFO("%s: %s workspace reserve took %.2f ms, sched copies = %d\n",
                __func__, n_tokens == n_tokens_tg ? "decode" : "prefill",
                (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
    } else {
        LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
                __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
    }
}

bool llama_context::initialize_moe_profile() {
    if (source_core_enabled()) { return !moe_profile_failed; }
    return initialize_moe_placement(moe_profiles, moe_statistics, moe_statistics_scores.empty() ? nullptr : moe_statistics_scores.data());
}

bool llama_context::initialize_moe_profile(const std::vector<ggml_backend_moe_static_profile_v1> & profiles) {
    return initialize_moe_placement(profiles, {});
}

bool llama_context::initialize_moe_statistics(const std::vector<ggml_backend_moe_source_statistics_v1> & statistics, const double * const * scores) {
    return initialize_moe_placement({}, statistics, scores);
}

bool llama_context::restore_moe_learning() {
    const bool learned = std::any_of(moe_profile_statistics.sources.begin(), moe_profile_statistics.sources.end(),
        [](const llama_moe_profile_source_statistics & source) { return !source.heat.empty(); });
    if (!learned) { return true; }
    if (!source_core_enabled()) { return false; }
    std::unique_lock<std::timed_mutex> publication_lock(moe_source_publication_mutex, std::try_to_lock);
    if (!publication_lock.owns_lock()) { return false; }
    std::lock_guard<std::mutex> caller_lock(moe_source_mutex);
    if (moe_source_callers || moe_source_closed.load() || moe_source_poisoned.load() || moe_profile_failed ||
            !moe_source_graph_variants.empty() || ggml_backend_sched_moe_source_selected_v1(sched.get())) { return false; }
    try {
        std::vector<ggml_backend_moe_source_learning_v1> views;
        views.reserve(moe_profile_statistics.sources.size());
        for (const auto & source : moe_profile_statistics.sources) {
            if (source.heat.size() != source.counts.size() || source.usage.size() != source.counts.size() || source.prior.size() != source.counts.size()) { return false; }
            views.push_back({sizeof(ggml_backend_moe_source_learning_v1), 1,
                {source.tensor, source.counts.data(), source.observations, uint32_t(source.counts.size()), source.domain},
                source.heat.data(), source.usage.data(), source.prior.data(), source.windows});
        }
        if (!ggml_moe_source_learning_valid(views.data(), uint32_t(views.size()))) { return false; }
        std::vector<std::vector<ggml_backend_moe_source_learning_v1>> per_owner(backends.size());
        std::vector<ggml_backend_moe_learning_restore_v1_t> endpoints(backends.size(), nullptr);
        for (const auto & record : views) {
            if (!is_moe_cached_tensor(record.source.tensor)) { continue; }
            auto * device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(record.source.tensor->buffer));
            bool found = false;
            for (size_t i = 0; i < backends.size(); ++i) {
                if (!device || ggml_backend_get_device(backends[i].get()) != device) { continue; }
                auto * reg = ggml_backend_dev_backend_reg(device);
                endpoints[i] = reinterpret_cast<ggml_backend_moe_learning_restore_v1_t>(
                    ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_LEARNING_RESTORE_V1_PROC_NAME));
                if (!endpoints[i]) { return false; }
                per_owner[i].push_back(record); found = true; break;
            }
            if (!found) { return false; }
        }
        const uint64_t deadline = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()) + 5'000'000'000ull;
        for (size_t i = 0; i < backends.size(); ++i) {
            if (!per_owner[i].empty() && !endpoints[i](backends[i].get(), per_owner[i].data(), uint32_t(per_owner[i].size()), 0, deadline)) {
                moe_profile_failed = true;
                return false;
            }
        }
        LLAMA_LOG_INFO("moe-profile: restored learned sources=%zu state=counts,heat,prior,usage,windows boundary=before-execution\n", views.size());
        return true;
    } catch (...) { moe_profile_failed = true; return false; }
}

bool llama_context::snapshot_moe_learning(std::vector<uint8_t> & bytes, uint32_t timeout_ms) {
    return snapshot_moe_learning_contexts({this}, bytes, timeout_ms);
}

bool llama_context::snapshot_moe_learning_contexts(const std::vector<llama_context *> & contexts, std::vector<uint8_t> & bytes, uint32_t timeout_ms) {
    if (contexts.empty() || contexts.size() > 64 || !timeout_ms || timeout_ms > 5000) { return false; }
    auto participants = contexts;
    for (const auto * context : participants) {
        if (!context || !context->source_core_enabled() || &context->model != &participants.front()->model) { return false; }
    }
    std::sort(participants.begin(), participants.end(), std::less<llama_context *>{});
    if (std::adjacent_find(participants.begin(), participants.end()) != participants.end()) { return false; }
    const uint64_t deadline = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()) + uint64_t(timeout_ms) * 1'000'000ull;
    const auto expired = [&] { return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()) >= deadline; };
    std::vector<std::unique_lock<std::timed_mutex>> publication_locks;
    std::vector<std::unique_lock<std::mutex>> caller_locks;
    publication_locks.reserve(participants.size()); caller_locks.reserve(participants.size());
    for (auto * context : participants) {
        publication_locks.emplace_back(context->moe_source_publication_mutex, std::try_to_lock);
        if (!publication_locks.back().owns_lock()) { return false; }
        caller_locks.emplace_back(context->moe_source_mutex, std::try_to_lock);
        if (!caller_locks.back().owns_lock() || context->moe_source_callers || context->moe_source_closed.load() ||
                context->moe_source_poisoned.load() || context->moe_profile_failed || context->moe_candidate_refresh_pending || expired()) { return false; }
    }
    auto & primary = *contexts.front();
    auto & model = primary.model;
    ggml_backend_moe_source_owner_v1 owner = {};
    ggml_backend_moe_source_lease_v1 lease = {};
    lease.struct_size = sizeof(lease); lease.abi_version = GGML_BACKEND_MOE_SOURCE_OWNER_V1_VERSION;
    if (!model.moe_source_owner_v1(&owner) || !owner.retain || !owner.release ||
            owner.retain(&owner, owner.generation, &lease) != GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK) { return false; }
    bool ok = false;
    std::vector<uint8_t> payload;
    try {
        auto snapshot = llama_moe_profile_learning_baseline(model.moe_sources(), primary.moe_profile_statistics, primary.moe_profiles);
        struct collection {
            llama_moe_profile_statistics * snapshot;
            std::vector<uint8_t> covered;
        } result = {&snapshot, std::vector<uint8_t>(snapshot.sources.size(), 0)};
        const auto collect = [](const ggml_backend_moe_source_learning_v1 * records, uint32_t count, void * data) -> bool {
            auto & result = *static_cast<collection *>(data);
            return llama_moe_profile_learning_overlay(*result.snapshot, result.covered, records, count);
        };
        for (auto * context : participants) {
            auto & backends = context->backends;
            std::vector<std::vector<ggml_backend_moe_source_identity_v1>> per_owner(backends.size());
            std::vector<ggml_backend_moe_learning_snapshot_v1_t> endpoints(backends.size(), nullptr);
            for (const auto & source : snapshot.sources) {
                if (!is_moe_cached_tensor(source.tensor)) { continue; }
                auto * device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(source.tensor->buffer));
                bool found = false;
                for (size_t i = 0; i < backends.size(); ++i) {
                    if (!device || ggml_backend_get_device(backends[i].get()) != device) { continue; }
                    auto * reg = ggml_backend_dev_backend_reg(device);
                    endpoints[i] = reinterpret_cast<ggml_backend_moe_learning_snapshot_v1_t>(
                        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_LEARNING_SNAPSHOT_V1_PROC_NAME));
                    if (!endpoints[i]) { throw std::runtime_error("learning snapshot backend unavailable"); }
                    per_owner[i].push_back({source.tensor, source.domain}); found = true; break;
                }
                if (!found) { throw std::runtime_error("learning snapshot source owner unavailable"); }
            }
            for (size_t i = 0; i < backends.size(); ++i) {
                if (expired() || (!per_owner[i].empty() && !endpoints[i](backends[i].get(), per_owner[i].data(), uint32_t(per_owner[i].size()),
                        0, deadline, collect, &result))) { throw std::runtime_error("learning snapshot boundary or history conflict"); }
            }
        }
        payload = llama_moe_profile_statistics_serialize(snapshot, model.moe_sources());
        ok = !expired();
    } catch (...) { ok = false; }
    const bool released = owner.release(&lease) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK;
    if (!released) { for (auto * context : participants) { context->moe_source_poisoned.store(true); } }
    if (!ok || !released || expired()) { return false; }
    bytes.swap(payload);
    return true;
}

bool llama_context::initialize_moe_placement(const std::vector<ggml_backend_moe_static_profile_v1> & profiles,
        const std::vector<ggml_backend_moe_source_statistics_v1> & statistics, const double * const * scores, bool source_preload) {
    if (profiles.empty() && statistics.empty()) { return scores == nullptr; }
    if (profiles.size() > GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS || (!profiles.empty() && !statistics.empty()) ||
            statistics.size() > GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS * GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS ||
            !ggml_moe_source_scores_valid(statistics.data(), scores, uint32_t(statistics.size()))) { return false; }
    std::unique_lock<std::timed_mutex> preload_publication_lock(moe_source_publication_mutex, std::defer_lock);
    std::unique_lock<std::mutex> preload_caller_lock(moe_source_mutex, std::defer_lock);
    if (source_preload) {
        if (!source_core_enabled() || !preload_publication_lock.try_lock()) { return false; }
        preload_caller_lock.lock();
        if (moe_source_callers || moe_source_closed.load() || moe_source_poisoned.load() || moe_profile_failed ||
                !moe_source_graph_variants.empty() || ggml_backend_sched_moe_source_selected_v1(sched.get())) { return false; }
    }
    if (source_core_enabled() && !source_preload) {
        std::unique_lock<std::timed_mutex> publication_lock(moe_source_publication_mutex, std::try_to_lock);
        if (!publication_lock.owns_lock()) { return false; }
        std::lock_guard<std::mutex> caller_lock(moe_source_mutex);
        if (moe_source_callers || moe_source_closed.load() || moe_source_poisoned.load() || moe_profile_failed ||
                !moe_source_graph_variants.empty() || ggml_backend_sched_moe_source_selected_v1(sched.get())) { return false; }
        try {
            std::map<uint32_t, std::unordered_set<const ggml_tensor *>> known;
            size_t expected = 0;
            for (const auto & source : model.moe_sources()) {
                for (const auto & bank : source.banks) {
                    if (bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE && bank.tensor &&
                            known[source.domain].insert(bank.tensor).second) { ++expected; }
                }
            }
            const auto known_tensor = [&](const ggml_tensor * tensor) {
                return std::any_of(known.begin(), known.end(), [&](const auto & domain) { return domain.second.count(tensor) != 0; });
            };
            std::vector<std::vector<int32_t>> ranks;
            std::vector<ggml_backend_moe_static_profile_v1> profile_views;
            std::unordered_set<const ggml_tensor *> ranked;
            uint64_t rank_count = 0;
            ranks.reserve(profiles.size()); profile_views.reserve(profiles.size());
            for (const auto & profile : profiles) {
                if (!profile.down || !known_tensor(profile.down) || !profile.experts || !profile.n_experts ||
                        profile.down->ne[2] <= 0 || profile.down->ne[2] > 65536 ||
                        profile.n_experts > uint64_t(profile.down->ne[2]) ||
                        profile.n_experts > (1u << 22) - rank_count || !ranked.insert(profile.down).second) { return false; }
                rank_count += profile.n_experts;
                std::vector<uint8_t> seen(profile.down->ne[2], 0);
                for (uint32_t index = 0; index < profile.n_experts; ++index) {
                    const int32_t expert = profile.experts[index];
                    if (expert < 0 || expert >= profile.down->ne[2] || seen[expert]++) { return false; }
                }
                ranks.emplace_back(profile.experts, profile.experts + profile.n_experts);
                profile_views.push_back({profile.down, ranks.back().data(), profile.n_experts});
            }
            llama_moe_profile_statistics owned;
            std::vector<ggml_backend_moe_source_statistics_v1> statistic_views;
            std::vector<const double *> score_views;
            owned.sources.reserve(statistics.size()); statistic_views.reserve(statistics.size());
            if (!statistics.empty() && statistics.size() != expected) { return false; }
            for (const auto & source : statistics) {
                const auto domain = known.find(source.domain);
                if (domain == known.end() || !domain->second.count(source.tensor)) { return false; }
                owned.sources.push_back({source.tensor, source.domain, source.observations,
                    std::vector<uint64_t>(source.counts, source.counts + source.n_experts)});
                if (scores) {
                    auto & bound = owned.sources.back();
                    const auto * first = scores[owned.sources.size() - 1];
                    bound.scores.assign(first, first + source.n_experts);
                    score_views.push_back(bound.scores.data());
                }
                statistic_views.push_back({source.tensor, owned.sources.back().counts.data(), source.observations, source.n_experts, source.domain});
            }
            owned.provenance = "Caller-supplied statistics validated against the canonical source catalog";
            moe_profile_ranks = std::move(ranks);
            moe_profiles = std::move(profile_views);
            moe_profile_statistics = std::move(owned);
            moe_statistics = std::move(statistic_views);
            moe_statistics_scores = std::move(score_views);
            LLAMA_LOG_INFO("moe-profile: stored source configuration ranks=%zu statistics=%zu application=next-graph-prepare\n",
                moe_profiles.size(), moe_statistics.size());
            return true;
        } catch (...) { return false; }
    }
    if (moe_profile_failed || (!source_preload && cparams.n_seq_max != 1)) {
        LLAMA_LOG_ERROR("moe-profile: GPU initialization requires a single quiescent request\n");
        return false;
    }
    synchronize();
    refresh_moe_candidates();
    std::vector<std::vector<ggml_backend_moe_static_profile_v1>> per_owner(backends.size());
    std::vector<ggml_backend_moe_profile_initialize_v1_t> endpoints(backends.size(), nullptr);
    std::vector<std::vector<ggml_backend_moe_source_statistics_v1>> per_owner_statistics(backends.size());
    std::vector<std::vector<const double *>> per_owner_scores(backends.size());
    std::vector<ggml_backend_moe_statistics_initialize_v2_t> scored_endpoints(backends.size(), nullptr);
    std::vector<ggml_backend_moe_statistics_initialize_v1_t> statistics_endpoints(backends.size(), nullptr);
    for (const auto & source : statistics) {
        if (!source.tensor || !source.tensor->buffer || !source.counts || !source.n_experts) { return false; }
        if (!is_moe_cached_tensor(source.tensor)) { continue; }
        auto * device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(source.tensor->buffer));
        bool found = false;
        for (size_t i = 0; i < backends.size(); ++i) {
            if (!device || ggml_backend_get_device(backends[i].get()) != device) { continue; }
            auto * reg = ggml_backend_dev_backend_reg(device);
            statistics_endpoints[i] = reinterpret_cast<ggml_backend_moe_statistics_initialize_v1_t>(
                ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_STATISTICS_INITIALIZE_V1_PROC_NAME));
            if (!statistics_endpoints[i]) { return false; }
            per_owner_statistics[i].push_back(source);
            if (scores) {
                scored_endpoints[i] = reinterpret_cast<ggml_backend_moe_statistics_initialize_v2_t>(
                    ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_STATISTICS_INITIALIZE_V2_PROC_NAME));
                if (!scored_endpoints[i]) { return false; }
                per_owner_scores[i].push_back(scores[size_t(&source - statistics.data())]);
            } found = true; break;
        }
        if (!found) { return false; }
    }
    for (const auto & profile : profiles) {
        if (!profile.down || !profile.down->buffer || !profile.experts || !profile.n_experts) { return false; }
        if (source_preload && !is_moe_cached_tensor(profile.down)) { continue; }
        auto * buffer_type = ggml_backend_buffer_get_type(profile.down->buffer);
        auto * device = ggml_backend_buft_get_device(buffer_type);
        if (!device) { return false; }
        bool found = false;
        for (size_t i = 0; i < backends.size(); ++i) {
            auto * backend = backends[i].get();
            if (ggml_backend_get_device(backend) != device) { continue; }
            auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
            endpoints[i] = reinterpret_cast<ggml_backend_moe_profile_initialize_v1_t>(
                ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_PROFILE_INITIALIZE_V1_PROC_NAME));
            if (!endpoints[i]) { return false; }
            per_owner[i].push_back(profile);
            found = true;
            break;
        }
        if (!found) { return false; }
    }
    ggml_backend_moe_source_owner_v1 owner = {};
    ggml_backend_moe_source_lease_v1 lease = {};
    lease.struct_size = sizeof(lease);
    lease.abi_version = GGML_BACKEND_MOE_SOURCE_OWNER_V1_VERSION;
    if (!model.moe_source_owner_v1(&owner) || !owner.retain || !owner.release ||
            owner.retain(&owner, owner.generation, &lease) != GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK) { return false; }
    const int64_t start = ggml_time_us();
    uint64_t total = 0;
    bool ok = true;
    const uint32_t flags = source_preload ? GGML_BACKEND_MOE_PLACEMENT_SOURCE_V1 : 0;
    for (size_t i = 0; i < backends.size(); ++i) {
        if (per_owner[i].empty() && per_owner_statistics[i].empty()) { continue; }
        uint64_t bytes = 0;
        const bool applied = per_owner_statistics[i].empty() ?
            endpoints[i](backends[i].get(), per_owner[i].data(), per_owner[i].size(), flags, &bytes) :
            scores ? scored_endpoints[i](backends[i].get(), per_owner_statistics[i].data(), per_owner_scores[i].data(), per_owner_statistics[i].size(), flags, &bytes) :
            statistics_endpoints[i](backends[i].get(), per_owner_statistics[i].data(), per_owner_statistics[i].size(), flags, &bytes);
        if (!applied) { ok = false; break; }
        total += bytes;
    }
    const bool released = owner.release(&lease) == GGML_BACKEND_MOE_SOURCE_STATUS_V1_OK;
    moe_profile_failed = !ok || !released;
    fprintf(stderr, "moe-profile: GPU update ranks=%zu statistics=%zu copied_bytes=%llu wall_us=%lld success=%d source_preload=%d policy=frequency-aware live replacement\n",
        profiles.size(), statistics.size(), (unsigned long long) total, (long long) (ggml_time_us() - start), !moe_profile_failed, int(source_preload));
    return !moe_profile_failed;
}

bool llama_moe_profile_initialize(llama_context * ctx) {
    return ctx && ctx->initialize_moe_profile();
}

extern "C" bool llama_moe_profile_snapshot_contexts(llama_context * const * contexts, size_t count, uint32_t timeout_ms,
        llama_moe_profile_write_callback write, void * user_data) {
    if (!contexts || !count || count > 64 || !write) { return false; }
    try {
        std::vector<uint8_t> bytes;
        return llama_context::snapshot_moe_learning_contexts({contexts, contexts + count}, bytes, timeout_ms) && write(bytes.data(), bytes.size(), user_data);
    } catch (...) { return false; }
}

extern "C" bool llama_moe_profile_snapshot(llama_context * ctx, uint32_t timeout_ms,
        llama_moe_profile_write_callback write, void * user_data) {
    return llama_moe_profile_snapshot_contexts(&ctx, 1, timeout_ms, write, user_data);
}

void llama_context::plan_moe_profile_capacities(enum llama_context_type ctx_type, const uint8_t * profile_data, size_t profile_bytes) {
    const auto & sources = model.moe_sources();
    const auto & memory = model.moe_expert_cache_group_memory(ctx_type);
    if (memory.size() != sources.size()) { throw std::runtime_error("MoE cache capacity geometry mismatch"); }
    std::vector<std::vector<long double>> strp_priorities(moe_profile_ranks.size());
    if (!moe_profile_ranks.empty()) {
        if (profile_bytes < 24 || memcmp(profile_data, "STRP", 4)) { throw std::runtime_error("MoE capacity profile format mismatch"); }
        for (size_t layer = 0; layer < strp_priorities.size(); ++layer) { strp_priorities[layer].reserve(moe_profile_ranks[layer].size()); }
        const uint32_t pairs = uint32_t(profile_data[20]) | uint32_t(profile_data[21]) << 8 |
            uint32_t(profile_data[22]) << 16 | uint32_t(profile_data[23]) << 24;
        if (pairs > (profile_bytes - 24) / 4) { throw std::runtime_error("MoE capacity profile pair extent mismatch"); }
        for (uint32_t i = 0; i < pairs; ++i) {
            const size_t offset = 24 + size_t(i) * 4;
            const uint32_t layer = uint32_t(profile_data[offset]) | uint32_t(profile_data[offset + 1]) << 8;
            if (layer >= strp_priorities.size()) { throw std::runtime_error("MoE capacity profile layer mismatch"); }
            strp_priorities[layer].push_back(pairs - i);
        }
    }
    struct owner_plan {
        uint64_t budget = 0;
        std::vector<ggml_moe_profile_capacity_group> groups;
        std::vector<const ggml_tensor *> down;
    };
    std::map<ggml_backend_dev_t, owner_plan> owners;
    for (size_t i = 0; i < sources.size(); ++i) {
        const auto & source = sources[i];
        const auto & cost = memory[i];
        if (!cost.per_slot_device_bytes) { continue; }
        const auto owner = model.moe_expert_cache_group_owner(i);
        if (!owner) { throw std::runtime_error("MoE capacity profile has no accelerator owner"); }
        const auto reg = ggml_backend_dev_backend_reg(owner);
        if (!reg || !ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CANDIDATE_REPLACE_CAPACITIES_V1_PROC_NAME)) {
            throw std::runtime_error("MoE backend does not support profile capacities; select uniform allocation");
        }
        const auto uniform = model.moe_expert_cache_slots(owner);
        if (uniform <= 0 || !cost.max_slots) { throw std::runtime_error("MoE uniform capacity has invalid group geometry"); }
        const ggml_tensor * down = nullptr;
        std::vector<ggml_moe_profile_bank_statistics> banks;
        for (const auto & bank : source.banks) {
            if (bank.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT ||
                    (source.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_ROUTED_MATRIX &&
                     bank.role == GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_ROUTED_WEIGHT)) { down = bank.tensor; }
            if (moe_profile_ranks.empty() && bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                const llama_moe_profile_source_statistics * statistic = nullptr;
                for (const auto & value : moe_profile_statistics.sources) {
                    if (value.tensor == bank.tensor && value.domain == source.domain) { statistic = &value; break; }
                }
                if (!statistic || !bank.tensor) { throw std::runtime_error("MoE capacity profile misses a routed source"); }
                banks.push_back({statistic->counts.data(), statistic->observations, bank.tensor->nb[2],
                    uint32_t(statistic->counts.size()), statistic->scores.empty() ? nullptr : statistic->scores.data()});
            }
        }
        if (!down || down->ne[2] <= 0 || uint64_t(down->ne[2]) != cost.max_slots) { throw std::runtime_error("MoE capacity profile has invalid down-bank geometry"); }
        ggml_moe_profile_capacity_group group;
        group.per_slot_bytes = cost.per_slot_device_bytes;
        group.minimum_slots = std::min({uint32_t(uniform), cost.max_slots, std::max(1u, model.hparams.n_expert_used(source.layer))});
        if (!moe_profile_ranks.empty()) {
            if (source.layer < 0 || size_t(source.layer) >= strp_priorities.size()) { throw std::runtime_error("MoE capacity profile has no layer priority"); }
            group.priorities = strp_priorities[source.layer];
        } else {
            std::vector<int32_t> ranks;
            std::vector<long double> scores;
            if (!ggml_moe_source_score_statistics(banks, scores) || !ggml_moe_source_rank_statistics(banks, ranks)) {
                throw std::runtime_error("MoE capacity profile statistics are invalid");
            }
            for (const auto expert : ranks) { group.priorities.push_back(scores[expert] / group.per_slot_bytes); }
        }
        if (group.priorities.size() < std::min(uint32_t(uniform), cost.max_slots) || group.priorities.size() > cost.max_slots) {
            throw std::runtime_error("MoE capacity profile has insufficient or excess ranks");
        }
        auto & plan = owners[owner];
        if (uint32_t(uniform) > (UINT64_MAX - plan.budget) / group.per_slot_bytes) { throw std::overflow_error("MoE capacity budget overflow"); }
        plan.budget += uint32_t(uniform) * group.per_slot_bytes;
        plan.down.push_back(down);
        plan.groups.push_back(std::move(group));
    }
    for (const auto & owner : owners) {
        bool observed = false;
        for (const auto & group : owner.second.groups) {
            observed |= !group.priorities.empty() && group.priorities.front() > 0;
        }
        if (!observed) {
            LLAMA_LOG_INFO("moe-cache-capacity: device=%s policy=uniform reason=no-profile-observations\n", ggml_backend_dev_name(owner.first));
            continue;
        }
        std::vector<uint32_t> capacities;
        uint64_t paid = 0;
        if (!ggml_moe_profile_plan_capacities(owner.second.groups, owner.second.budget, capacities, paid)) {
            throw std::runtime_error("MoE profile cannot fit fixed group capacities within the device budget");
        }
        uint32_t minimum = UINT32_MAX, maximum = 0;
        uint64_t total = 0;
        for (size_t i = 0; i < capacities.size(); ++i) {
            if (!moe_profile_capacities.emplace(owner.second.down[i], capacities[i]).second) {
                throw std::runtime_error("MoE capacity profile has ambiguous group identity");
            }
            minimum = std::min(minimum, capacities[i]); maximum = std::max(maximum, capacities[i]); total += capacities[i];
        }
        LLAMA_LOG_INFO("moe-cache-capacity: device=%s policy=profile groups=%zu slots=%llu min=%u max=%u variable_bytes=%llu budget_bytes=%llu fixed_costs=unchanged\n",
            ggml_backend_dev_name(owner.first), capacities.size(), (unsigned long long) total, minimum, maximum,
            (unsigned long long) paid, (unsigned long long) owner.second.budget);
    }
}

void llama_context::refresh_moe_candidates() {
    if (!moe_candidate_refresh_pending) {
        return;
    }

    moe_candidate_refresh_pending = false;
    if (moe_candidate_replace_fns.empty()) {
        return;
    }

    ggml_backend_moe_candidate_snapshot_v2 disabled = {};
    disabled.magic = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_MAGIC;
    disabled.abi_version = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_VERSION;
    disabled.struct_size = sizeof(disabled);
    disabled.n_slots = std::max(model.moe_expert_cache_slots(), 0);

    try {
        const llama_moe_candidate_snapshot candidates(model, *loras);
        for (const auto & endpoint : moe_candidate_replace_fns) {
            auto owner_candidates = candidates.get();
            owner_candidates.n_slots = std::max(
                model.moe_expert_cache_slots(ggml_backend_get_device(endpoint.first)), 0);
            int32_t result;
            if (moe_profile_capacities.empty()) {
                result = endpoint.second(endpoint.first, &owner_candidates);
            } else {
                const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(endpoint.first));
                const auto replace = reinterpret_cast<ggml_backend_moe_candidate_replace_capacities_v1_t>(
                    ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CANDIDATE_REPLACE_CAPACITIES_V1_PROC_NAME));
                if (!replace) { throw std::runtime_error("MoE backend does not support profile capacities; select uniform allocation"); }
                std::vector<uint32_t> capacities(owner_candidates.n_groups, owner_candidates.n_slots);
                for (uint32_t i = 0; i < owner_candidates.n_tensors; ++i) {
                    const auto & tensor = owner_candidates.tensors[i];
                    const auto found = moe_profile_capacities.find(tensor.tensor);
                    if (found != moe_profile_capacities.end() && tensor.group_index < capacities.size()) { capacities[tensor.group_index] = found->second; }
                }
                result = replace(endpoint.first, &owner_candidates, capacities.data(), capacities.size());
            }
            if (result != GGML_BACKEND_MOE_CANDIDATE_REPLACE_ACCEPTED &&
                    result != GGML_BACKEND_MOE_CANDIDATE_REPLACE_REJECTED) {
                auto owner_disabled = disabled;
                owner_disabled.n_slots = owner_candidates.n_slots;
                endpoint.second(endpoint.first, &owner_disabled);
            }
        }
    } catch (...) {
        if (!moe_profile_capacities.empty()) {
            moe_profile_failed = true;
            LLAMA_LOG_ERROR("moe-cache-capacity: candidate publication failed; grouped execution disabled\n");
        }
        for (const auto & endpoint : moe_candidate_replace_fns) {
            endpoint.second(endpoint.first, &disabled);
        }
    }
}

void llama_context::refresh_moe_layer_owners() {
    if (model.moe_expert_cache_slots() <= 0) {
        return;
    }
    std::vector<ggml_backend_t> owners(model.layers.size(), nullptr);
    std::unordered_map<ggml_backend_buffer_type_t, bool> cached_types;
    std::unordered_set<ggml_backend_t> checked;
    bool                        participating = false;
    bool                        supported     = true;
    for (const auto & use : model.tensor_uses()) {
        if (use.layer < 0 || size_t(use.layer) >= owners.size() || !use.tensor || !use.tensor->buffer) {
            continue;
        }
        const auto buft = ggml_backend_buffer_get_type(use.tensor->buffer);
        auto cached = cached_types.emplace(buft, false);
        if (cached.second) {
            cached.first->second = is_moe_cached_tensor(use.tensor);
        }
        if (!cached.first->second) {
            continue;
        }
        participating = true;
        if (!owners[use.layer]) {
            for (const auto & backend : backends) {
                if (ggml_backend_get_device(backend.get()) == model.dev_layer(use.layer)) {
                    owners[use.layer] = backend.get();
                    break;
                }
            }
        }
        if (supported && checked.insert(owners[use.layer]).second) {
            supported = backend_supports_required_grouped_execution(owners[use.layer]);
        }
    }
    const bool capability_changed            = moe_required_grouped_execution_supported != (participating && supported);
    moe_required_grouped_execution_supported = participating && supported;
    if (owners == moe_layer_owners && !capability_changed) {
        return;
    }
    if (sched) {
        if (reset_source_core_checked() != GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK) { return; }
        ggml_backend_sched_synchronize(sched.get());
        ggml_backend_sched_reset(sched.get());
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        gf_res_prev_active = nullptr;
        if (gf_res_reserve) {
            gf_res_reserve->reset();
        }
        ++graph_execution_owner_generation;
    }
    moe_layer_owners   = std::move(owners);
    request_sched_reserve();
}

void llama_context::place_moe_regions(llm_graph_result * res) {
    if (model.moe_expert_cache_slots() <= 0 || model.split_mode() != LLAMA_SPLIT_MODE_LAYER ||
            (!moe_hybrid_metadata && !loras->empty())) {
        return;
    }
    const auto & sources = model.moe_sources();
    for (auto & region : res->get_moe_regions()) {
        if (region.layer < 0 || size_t(region.layer) >= moe_layer_owners.size()) {
            continue;
        }
        uint32_t semantic_group = UINT32_MAX;
        uint32_t domain = 0;
        bool valid = true;
        for (auto * node : region.body_operations) {
            if (!node || node->op != GGML_OP_MUL_MAT_ID || !is_moe_cached_tensor(node->src[0])) {
                continue;
            }
            bool matched = false;
            for (uint32_t index = 0; index < sources.size() && index < GGML_BACKEND_MOE_CANDIDATE_MAX_GROUPS; ++index) {
                const auto & source = sources[index];
                if (source.layer != region.layer || source.layout == GGML_BACKEND_MOE_CANDIDATE_LAYOUT_INVALID) {
                    continue;
                }
                for (const auto & bank : source.banks) {
                    if (bank.tensor != node->src[0] || bank.status != GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                        continue;
                    }
                    if (matched || (semantic_group != UINT32_MAX && domain != source.domain)) { valid = false; }
                    if (semantic_group == UINT32_MAX) { semantic_group = index; domain = source.domain; }
                    matched = true;
                }
            }
            if (!matched) { valid = false; }
            if (!valid) { break; }
        }
        auto * owner = moe_layer_owners[region.layer];
        if (!valid || semantic_group == UINT32_MAX || !backend_supports_required_grouped_execution(owner)) {
            continue;
        }
        const bool placed = region.place(sched.get(), owner);
        if (placed) { region.semantic_group = semantic_group; region.domain = domain; }
        LLAMA_LOG_INFO("moe-placement: layer=%d semantic_group=%u domain=%u owner=%s placed=%d operations=%zu\n",
                       region.layer, semantic_group, domain, ggml_backend_name(owner), placed,
                       region.operations.size());
    }
}


static bool measure_moe_regions(const llm_graph_result & graph, bool source_core, uint32_t & count, uint32_t & prepared_count) {
    const auto & regions = graph.get_moe_regions();
    if (regions.size() > UINT32_MAX) {
        LLAMA_LOG_ERROR("moe-hybrid: region count exceeds the scheduler interface\n");
        return false;
    }
    uint64_t prepared = 0;
    count = 0;
    for (const auto & region : regions) {
        if (region.route == nullptr || region.route->ne[0] <= 0 || uint64_t(region.route->ne[0]) > UINT32_MAX ||
                region.route->ne[1] <= 0 || uint64_t(region.route->ne[1]) > UINT32_MAX ||
                uint64_t(region.route->ne[0]) > UINT32_MAX / uint64_t(region.route->ne[1])) {
            LLAMA_LOG_ERROR("moe-hybrid: route geometry exceeds the scheduler interface layer=%d\n", region.layer);
            return false;
        }
        const uint64_t projections = source_core ? std::max<size_t>(1,
            std::count_if(region.body_operations.begin(), region.body_operations.end(),
                [](const ggml_tensor * node) { return node && node->op == GGML_OP_MUL_MAT_ID && !ggml_is_empty(node); })) : 1;
        const uint64_t region_capacity = std::max<uint64_t>(region.route->ne[0] + 1, projections);
        if (projections > UINT32_MAX || region_capacity > UINT32_MAX || count > UINT32_MAX - projections || prepared > UINT32_MAX - region_capacity) {
            LLAMA_LOG_ERROR("moe-hybrid: prepared region capacity exceeds the scheduler interface layer=%d\n", region.layer);
            return false;
        }
        count += projections;
        prepared += region_capacity;
    }
    prepared_count = prepared;
    return true;
}

bool llama_context::finalize_moe_regions(llm_graph_result * res, bool hybrid_decode,
        const ggml_graph_execution_certificate * certificate) {
    const bool required_prefill = moe_hybrid_required && certificate &&
        certificate->domain == GGML_GRAPH_EXECUTION_DOMAIN_MAIN &&
        (certificate->row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL ||
         certificate->row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT) && certificate->n_rows > 1;
    std::unique_lock<std::timed_mutex> publication_lock(moe_source_publication_mutex, std::defer_lock);
    if (source_core_enabled()) {
        publication_lock.lock();
        if (moe_source_closed.load() || moe_source_poisoned.load()) { return false; }
    }
    if (!moe_hybrid_metadata) {
        return true;
    }
    if (moe_hybrid_required && ((!source_core_enabled() && (cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || cparams.n_seq_max != 1)) ||
            cparams.pipeline_parallel || cparams.cb_eval != nullptr || (!source_core_enabled() && sched_buffer_owner != nullptr) ||
            res->get_moe_regions().empty())) {
        if (source_core_enabled()) {
            if (required_prefill) {
                LLAMA_LOG_ERROR("moe-hybrid: required prefill context or region is unsupported\n");
                return false;
            }
            if (!moe_profiles.empty() || !moe_statistics.empty()) { LLAMA_LOG_ERROR("moe-profile: required static placement is unsupported by this source context or region\n"); return false; }
            if (ggml_backend_sched_moe_source_fallback_v1(sched.get())) { moe_source_poisoned.store(true); return false; }
            LLAMA_LOG_INFO("moe-source-core-admission: provider=normal reason=context-capability effects_started=0\n");
            return true;
        }
        LLAMA_LOG_ERROR("moe-hybrid: unsupported context or missing MoE regions\n");
        return false;
    }
    if (moe_hybrid_required && !hybrid_decode) {
        return true;
    }
    if (hybrid_decode && source_core_enabled()) {
        for (const auto & region : res->get_moe_regions()) {
            auto * reg = region.backend ? ggml_backend_dev_backend_reg(ggml_backend_get_device(region.backend)) : nullptr;
            if (!reg || !ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_SOURCE_CORE_V1_PROC_NAME)) {
                if (required_prefill) {
                    LLAMA_LOG_ERROR("moe-hybrid: required prefill backend is unsupported layer=%d\n", region.layer);
                    return false;
                }
                if (!moe_profiles.empty() || !moe_statistics.empty()) { LLAMA_LOG_ERROR("moe-profile: required static placement is unsupported by this source context or region\n"); return false; }
                if (ggml_backend_sched_moe_source_fallback_v1(sched.get())) { moe_source_poisoned.store(true); return false; }
                LLAMA_LOG_INFO("moe-source-core-admission: provider=normal reason=backend-capability effects_started=0\n");
                return true;
            }
        }
    }
    uint64_t allocator_generation = 0;
    uint64_t shrink_generation = 0;
    ggml_backend_sched_get_buffer_state(sched.get(), &allocator_generation, &shrink_generation);
    if (allocator_generation == 0) {
        LLAMA_LOG_ERROR("moe-hybrid-metadata: allocator generation unavailable; enable phase-aware workspace\n");
        return false;
    }
    uint32_t hybrid_region_count = 0;
    uint32_t hybrid_prepared_region_count = 0;
    if (hybrid_decode) {
        if (!measure_moe_regions(*res, source_core_enabled(), hybrid_region_count, hybrid_prepared_region_count)) { return false; }
        if (source_core_enabled() && gf_res_reserve) {
            uint32_t reserved_count = 0, reserved_prepared = 0;
            if (!measure_moe_regions(*gf_res_reserve, true, reserved_count, reserved_prepared)) { return false; }
            hybrid_region_count = std::max(hybrid_region_count, reserved_count);
            hybrid_prepared_region_count = std::max(hybrid_prepared_region_count, reserved_prepared);
        }
        if (source_core_enabled()) {
            moe_source_region_capacity = std::max(moe_source_region_capacity, hybrid_region_count);
            moe_source_prepared_capacity = std::max(moe_source_prepared_capacity, hybrid_prepared_region_count);
            hybrid_region_count = moe_source_region_capacity;
            hybrid_prepared_region_count = moe_source_prepared_capacity;
        }
    }
    for (auto & region : res->get_moe_regions()) {
        if (!region.builder_cut_closed || region.has_lora) {
            region.finalized_metadata.reset();
            if (hybrid_decode) {
                if (source_core_enabled()) {
                    if (required_prefill) {
                        LLAMA_LOG_ERROR("moe-hybrid: required prefill expert cut is unsupported layer=%d\n", region.layer);
                        return false;
                    }
                    if (!moe_profiles.empty() || !moe_statistics.empty()) { LLAMA_LOG_ERROR("moe-profile: required static placement is unsupported by this source context or region\n"); return false; }
                    if (ggml_backend_sched_moe_source_fallback_v1(sched.get())) { moe_source_poisoned.store(true); return false; }
                    LLAMA_LOG_INFO("moe-source-core-admission: provider=normal reason=expert-cut-capability layer=%d effects_started=0\n", region.layer);
                    return true;
                }
                LLAMA_LOG_ERROR("moe-hybrid: unsupported region cut layer=%d\n", region.layer);
                return false;
            }
            continue;
        }
        if (region.backend == nullptr) {
            for (auto & clear : res->get_moe_regions()) {
                clear.finalized_metadata.reset();
            }
            LLAMA_LOG_ERROR("moe-hybrid-metadata: required finalization failed layer=%d status=%d\n", region.layer,
                            GGML_BACKEND_SCHED_REGION_STATUS_V1_NOT_FINALIZED);
            return false;
        }
        const int32_t status = region.finalize_metadata(
            sched.get(), res->get_gf(), graph_execution_owner_generation, allocator_generation);
        if (status != GGML_BACKEND_SCHED_REGION_STATUS_V1_OK) {
            for (auto & clear : res->get_moe_regions()) {
                clear.finalized_metadata.reset();
            }
            LLAMA_LOG_ERROR("moe-hybrid-metadata: required finalization failed layer=%d status=%d\n", region.layer,
                            status);
            return false;
        }
        if (hybrid_decode) {
            ggml_backend_moe_source_owner_v1 owner = {};
            if ((!source_core_enabled() && (cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || cparams.n_seq_max != 1)) ||
                    cparams.pipeline_parallel || cparams.cb_eval != nullptr || (!source_core_enabled() && sched_buffer_owner != nullptr) ||
                    !model.moe_source_owner_v1(&owner)) {
                LLAMA_LOG_ERROR("moe-hybrid: unsupported context or source owner\n");
                return false;
            }
            ggml_backend_moe_hybrid_config_v1 config = {};
            config.struct_size = sizeof(config);
            config.n_threads = cparams.n_threads;
            config.max_regions = hybrid_region_count;
            config.max_prepared_regions = hybrid_prepared_region_count;
            if (source_core_enabled()) {
                const uint64_t variants = moe_source_max_programs;
                if (config.max_prepared_regions > UINT32_MAX / variants) { return false; }
                config.max_prepared_regions *= variants;
                config.max_source_programs = variants;
            }
            config.gpu_miss_quota = moe_hybrid_gpu_misses;
            config.admission_quota = moe_hybrid_demand_admission ? moe_hybrid_admission_misses : 0;
            config.demand_admission = moe_hybrid_demand_admission;
            config.resident_batch = moe_hybrid_resident_batch;
            config.executor = moe_hybrid_executor;
            config.cpu_flags = GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNKNOWN_THREAD_STACK_BYTES;
            if (moe_hybrid_allow_runtime_allocations) {
                config.cpu_flags |= GGML_BACKEND_MOE_CPU_SERVICE_FLAG_V1_ALLOW_UNPROVEN_RUNTIME_ALLOCATIONS;
            }
            config.backend = region.backend;
            config.source_owner = &owner;
            config.profiles = moe_profiles.empty() ? nullptr : moe_profiles.data();
            config.n_profiles = moe_profiles.size();
            config.statistics = moe_statistics.empty() ? nullptr : moe_statistics.data();
            config.n_statistics = moe_statistics.size();
            config.statistics_scores = moe_statistics_scores.empty() ? nullptr : moe_statistics_scores.data();
            config.profile_adaptation = moe_hybrid_profile_adapt;
            config.cpu_module_acquire = ggml_backend_moe_cpu_module_acquire_v1;
            config.module_retain = ggml_backend_moe_module_retain_v1;
            config.module_release = ggml_backend_moe_module_release_v1;
            int32_t prepared = ggml_backend_sched_moe_hybrid_configure_v1(sched.get(), &config);
            if (prepared == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
                prepared = region.prepare_hybrid(sched.get(), owner, config.n_threads, certificate, source_core_enabled());
            }
            if (prepared != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) {
                if (source_core_enabled() && (prepared == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION ||
                        prepared == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_PRECISION)) {
                    if (required_prefill) {
                        LLAMA_LOG_ERROR("moe-hybrid: required prefill operation is unsupported layer=%d status=%d\n", region.layer, prepared);
                        return false;
                    }
                    if (!moe_profiles.empty() || !moe_statistics.empty()) { LLAMA_LOG_ERROR("moe-profile: required static placement is unsupported by this source context or region\n"); return false; }
                    if (ggml_backend_sched_moe_source_fallback_v1(sched.get())) { moe_source_poisoned.store(true); return false; }
                    LLAMA_LOG_INFO("moe-source-core-admission: provider=normal reason=expert-trait-capability layer=%d status=%d effects_started=0\n",
                        region.layer, prepared);
                    return true;
                }
                LLAMA_LOG_ERROR("moe-hybrid: preparation failed layer=%d status=%d\n", region.layer, prepared);
                return false;
            }
        }
    }
    return true;
}

bool llama_context::moe_graph_supports_required_grouped(ggml_cgraph * gf) const {
    if (moe_source_graph_capacity && gf_res_prev_active && gf_res_prev_active->get_gf() == gf &&
            gf_res_prev_active->required_grouped_prepared) {
        const auto & owners = gf_res_prev_active->required_grouped_backends;
        for (auto * owner : owners) {
            if (!backend_supports_required_grouped_execution(owner)) { return false; }
        }
        return model.moe_expert_cache_slots() > 0 && !owners.empty();
    }
    return model.moe_expert_cache_slots() > 0 && graph_supports_required_grouped_execution(sched.get(), gf);
}

void llama_context::prepare_required_grouped_execution(llm_graph_result * res) {
    if (!moe_source_graph_capacity) { return; }
    auto * graph = res->get_gf();
    auto & owners = res->required_grouped_backends;
    owners.clear();
    if (model.moe_expert_cache_slots() > 0) {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (node->op != GGML_OP_MUL_MAT_ID || ggml_is_empty(node)) { continue; }
            auto * owner = ggml_backend_sched_get_tensor_backend(sched.get(), node);
            if (std::find(owners.begin(), owners.end(), owner) == owners.end()) { owners.push_back(owner); }
        }
    }
    res->required_grouped_prepared = true;
}

void llama_context::synchronize() {
    if (!sched) {
        workspace_in_flight = false;
        return;
    }

    ggml_backend_sched_synchronize(sched.get());
    workspace_in_flight = false;

    finish_compute(n_queued_tokens, ggml_time_us() - t_compute_start_us);
    n_queued_tokens = 0;
    t_compute_start_us = 0;
    ++compute_sync_generation;
}

bool llama_context::source_graph_cache_enabled() const {
    return source_core_enabled() && !graph_reuse_disable && cparams.phase_aware_workspace &&
        !cparams.pipeline_parallel && !cparams.cb_eval;
}

bool llama_context::source_core_enabled() const {
    return moe_hybrid_required && moe_hybrid_executor == GGML_BACKEND_MOE_HYBRID_EXECUTOR_V1_FIDELITY &&
        ggml_moe_fidelity_selection().source_pool;
}

bool llama_context::begin_source_call() {
    if (!source_core_enabled()) { return true; }
    std::lock_guard<std::mutex> lock(moe_source_mutex);
    if (moe_source_closed.load() || moe_source_poisoned.load() ||
            (moe_source_callers && moe_source_caller != std::this_thread::get_id())) { return false; }
    ++moe_source_callers;
    moe_source_caller = std::this_thread::get_id();
    return true;
}

void llama_context::end_source_call() {
    if (!source_core_enabled()) { return; }
    std::lock_guard<std::mutex> lock(moe_source_mutex);
    GGML_ASSERT(moe_source_callers && moe_source_caller == std::this_thread::get_id());
    if (!--moe_source_callers) { moe_source_caller = {}; }
    moe_source_condition.notify_all();
}

int32_t llama_context::reset_source_core_checked(bool all_variants) {
    if (!source_core_enabled()) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK; }
    std::unique_lock<std::timed_mutex> publication_lock(moe_source_publication_mutex, std::defer_lock);
    if (!publication_lock.try_lock_for(std::chrono::seconds(5))) {
        moe_source_poisoned.store(true);
        return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_TIMEOUT;
    }
    if (all_variants) {
        for (auto & variant : moe_source_graph_variants) {
            const auto status = ggml_backend_sched_moe_source_reset_v1(variant.scheduler.get());
            if (status) { moe_source_poisoned.store(true); return status; }
        }
        moe_source_graph_variants.clear();
    }
    if (ggml_backend_sched_moe_source_selected_v1(sched.get())) {
        const auto status = all_variants ? ggml_backend_sched_moe_source_reset_v1(sched.get()) :
            ggml_backend_sched_moe_source_reset_graph_v1(sched.get());
        if (status) { moe_source_poisoned.store(true); return status; }
    }
    if (all_variants) { moe_hybrid_graph_certificate = {}; }
    return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
}

int32_t llama_context::retire_source_sampler(llama_seq_id seq_id) {
    if (!source_core_enabled()) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK; }
    const auto references = [seq_id](const auto & graphs) {
        return std::any_of(graphs.begin(), graphs.end(), [seq_id](const auto & graph) {
            return graph && graph->references_sampler(seq_id);
        });
    };
    const bool active_references = references(gf_res_prev);
    const bool reserve_references = gf_res_reserve && gf_res_reserve->references_sampler(seq_id);
    if (!active_references && !reserve_references &&
            std::none_of(moe_source_graph_variants.begin(), moe_source_graph_variants.end(),
                [&](const auto & variant) { return references(variant.graphs); })) {
        return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
    }
    synchronize();
    std::unique_lock<std::timed_mutex> publication_lock(moe_source_publication_mutex, std::defer_lock);
    if (!publication_lock.try_lock_for(std::chrono::seconds(5))) {
        moe_source_poisoned.store(true);
        return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_TIMEOUT;
    }
    for (auto & variant : moe_source_graph_variants) {
        if (!references(variant.graphs)) { continue; }
        const auto status = ggml_backend_sched_moe_source_reset_graph_v1(variant.scheduler.get());
        if (status) { moe_source_poisoned.store(true); return status; }
        for (auto & graph : variant.graphs) { if (graph) { graph->reset(); } }
        variant.active = nullptr;
    }
    if (active_references || (reserve_references && !gf_res_prev_active)) {
        if (ggml_backend_sched_moe_source_selected_v1(sched.get())) {
            const auto status = ggml_backend_sched_moe_source_reset_graph_v1(sched.get());
            if (status) { moe_source_poisoned.store(true); return status; }
        } else { ggml_backend_sched_reset(sched.get()); }
        for (auto & graph : gf_res_prev) { if (graph) { graph->reset(); } }
        gf_res_prev_active = nullptr;
    }
    if (reserve_references) { gf_res_reserve->reset(); }
    return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
}

bool llama_context::select_source_graph_variant(const ggml_graph_execution_certificate & certificate) {
    const uint32_t rows = certificate.n_rows;
    if (!source_core_enabled()) { return true; }
    uint64_t generation = 0, shrink_generation = 0;
    ggml_backend_sched_get_buffer_state(sched.get(), &generation, &shrink_generation);
    const auto retirement_epoch = ggml_backend_sched_moe_source_retirement_epoch_v1(sched.get());
    if (generation != sched_buffer_generation || retirement_epoch != sched_source_retirement_epoch) {
        if (reset_source_core_checked(!source_graph_cache_enabled())) { return false; }
        for (auto & graph : gf_res_prev) { if (graph) { graph->reset(); } }
        gf_res_prev_active = nullptr;
        sched_buffer_generation = generation;
        sched_shrink_generation = shrink_generation;
        sched_source_retirement_epoch = retirement_epoch;
    }
    const uint32_t input_mode = use_sampled_input_async ? 2 : unsigned(use_sampled_input);
    if (!rows || rows > cparams.n_ubatch || !source_graph_cache_enabled()) {
        if (!moe_source_graph_variants.empty() && reset_source_core_checked()) { return false; }
        return true;
    }
    std::unique_lock<std::timed_mutex> publication_lock(moe_source_publication_mutex, std::defer_lock);
    if (!publication_lock.try_lock_for(std::chrono::seconds(5))) { moe_source_poisoned.store(true); return false; }
    if (moe_source_poisoned || moe_source_closed) { return false; }
    if (!memcmp(&moe_hybrid_graph_certificate, &certificate, sizeof(certificate)) &&
            moe_source_active_outputs == n_outputs &&
            moe_source_active_input_mode == input_mode) { return true; }
    const bool retain_active = ggml_backend_sched_moe_source_selected_v1(sched.get()) &&
        moe_hybrid_graph_certificate.n_rows;
    try {
        moe_source_graph_variants.reserve(moe_source_max_programs);
    } catch (const std::exception &) {
        moe_source_poisoned.store(true);
        return false;
    }
    auto found = std::find_if(moe_source_graph_variants.begin(), moe_source_graph_variants.end(),
        [&](const auto & variant) { return variant.outputs == n_outputs && variant.input_mode == input_mode &&
            !memcmp(&variant.certificate, &certificate, sizeof(certificate)); });
    if (!retain_active && found == moe_source_graph_variants.end()) { return true; }

    // Each capture keeps its scheduler splits and public tensor backing alive.
    const auto drained = ggml_backend_sched_moe_source_drain_v1(sched.get());
    if (drained) { moe_source_poisoned.store(true); return false; }
    ggml_backend_sched_synchronize(sched.get());
    workspace_in_flight = false;
    moe_source_graph_variant next;
    if (found == moe_source_graph_variants.end()) {
        if (moe_source_graph_variants.size() >= moe_source_max_programs - 1) {
            auto retired = std::move(moe_source_graph_variants.front());
            moe_source_graph_variants.erase(moe_source_graph_variants.begin());
            const auto status = ggml_backend_sched_moe_source_reset_v1(retired.scheduler.get());
            if (status) {
                moe_source_graph_variants.insert(moe_source_graph_variants.begin(), std::move(retired));
                moe_source_poisoned.store(true);
                return false;
            }
        }
        ggml_backend_sched_t clone = nullptr;
        const auto status = ggml_backend_sched_moe_source_clone_v1(sched.get(), &clone);
        if (status) { moe_source_poisoned.store(true); return false; }
        next.scheduler.reset(clone);
        ggml_backend_sched_get_buffer_state(clone, &next.buffer_generation, &next.shrink_generation);
    } else {
        next = std::move(*found);
        moe_source_graph_variants.erase(found);
        uint64_t next_generation = 0, next_shrink = 0;
        ggml_backend_sched_get_buffer_state(next.scheduler.get(), &next_generation, &next_shrink);
        const auto next_epoch = ggml_backend_sched_moe_source_retirement_epoch_v1(next.scheduler.get());
        if (next_generation != next.buffer_generation || next_epoch != next.retirement_epoch) {
            const auto status = ggml_backend_sched_moe_source_reset_graph_v1(next.scheduler.get());
            if (status) {
                moe_source_graph_variants.push_back(std::move(next));
                moe_source_poisoned.store(true);
                return false;
            }
            for (auto & graph : next.graphs) { if (graph) { graph->reset(); } }
            next.active = nullptr;
            next.certificate = {};
            next.buffer_generation = next_generation;
            next.shrink_generation = next_shrink;
            next.retirement_epoch = next_epoch;
        }
    }
    moe_source_graph_variant previous;
    previous.graphs = std::move(gf_res_prev);
    previous.scheduler = std::move(sched);
    previous.active = gf_res_prev_active;
    previous.certificate = moe_hybrid_graph_certificate;
    previous.outputs = moe_source_active_outputs;
    previous.sampled_device = sampled_inputs_device;
    previous.input_mode = moe_source_active_input_mode;
    previous.reserved_tokens = sched_reserved_tokens;
    previous.reserved_kv = sched_reserved_kv;
    previous.buffer_generation = sched_buffer_generation;
    previous.shrink_generation = sched_shrink_generation;
    previous.retirement_epoch = sched_source_retirement_epoch;
    if (retain_active) { moe_source_graph_variants.push_back(std::move(previous)); }
    gf_res_prev = std::move(next.graphs);
    sched = std::move(next.scheduler);
    gf_res_prev_active = next.active;
    moe_hybrid_graph_certificate = next.certificate;
    moe_source_active_outputs = next.outputs;
    sampled_inputs_device = next.sampled_device;
    moe_source_active_input_mode = next.input_mode;
    sched_reserved_tokens = next.reserved_tokens;
    sched_reserved_kv = next.reserved_kv;
    sched_buffer_generation = next.buffer_generation;
    sched_shrink_generation = next.shrink_generation;
    sched_source_retirement_epoch = next.retirement_epoch;
    LLAMA_LOG_DEBUG("moe-source-core-variants: selected_rows=%u sequences=%u domain=%u semantics=%u retained=%zu reused=%u\n",
        rows, certificate.n_sequences, certificate.domain, certificate.row_semantics,
        moe_source_graph_variants.size(), unsigned(next.active != nullptr));
    return true;
}

int32_t llama_context::close_source_core_checked() {
    if (!source_core_enabled()) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK; }
    const auto expiry = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    {
        std::lock_guard<std::mutex> lock(moe_source_mutex);
        if (moe_source_callers && moe_source_caller == std::this_thread::get_id()) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_BUSY; }
        moe_source_closed.store(true);
    }
    {
        std::unique_lock<std::timed_mutex> publication_lock(moe_source_publication_mutex, std::defer_lock);
        if (!publication_lock.try_lock_until(expiry)) { moe_source_poisoned.store(true); return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_TIMEOUT; }
        if (sched) {
            auto status = ggml_backend_sched_moe_source_close_v1(sched.get());
            if (status) { moe_source_poisoned.store(true); return status; }
            status = ggml_backend_sched_moe_source_drain_v1(sched.get());
            if (status) { moe_source_poisoned.store(true); return status; }
        }
    }
    {
        std::unique_lock<std::mutex> lock(moe_source_mutex);
        if (!moe_source_condition.wait_until(lock, expiry, [&] { return moe_source_callers == 0; })) {
            moe_source_poisoned.store(true);
            return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_TIMEOUT;
        }
    }
    return reset_source_core_checked();
}

struct llama_source_call_guard {
    llama_context & context;
    bool entered;
    explicit llama_source_call_guard(llama_context & context) : context(context), entered(context.begin_source_call()) {}
    ~llama_source_call_guard() { if (entered) { context.end_source_call(); } }
};

void llama_context::finish_compute(int64_t n_tokens, int64_t elapsed_us) {
    if (n_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += elapsed_us;
        }
        n_eval++;
    } else if (n_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += elapsed_us;
        }
        n_p_eval += n_tokens;
    }

    if (n_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }
}

uint64_t llama_context::trim_transient_memory() {
    if (!cparams.live_context_workspace) {
        return 0;
    }

    synchronize();

    using trim_fn = uint64_t (*)(ggml_backend_t);
    uint64_t released = 0;
    for (ggml_backend_t backend : backend_ptrs) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
        auto * fn = reg != nullptr ? (trim_fn) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_trim_transient_pools") : nullptr;
        if (fn != nullptr) {
            released += fn(backend);
        }
    }
    return released;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

bool llama_context::set_moe_test_hook(ggml_backend_moe_hybrid_test_hook_v1_t hook, void * data) {
    std::lock_guard<std::mutex> lock(moe_source_mutex);
    if (!source_core_enabled() || moe_source_callers || moe_source_closed.load() || moe_source_poisoned.load()) { return false; }
    moe_test_hook = hook;
    moe_test_hook_data = hook ? data : nullptr;
    moe_test_hook_set = true;
    return true;
}

const llama_moe_test_frame * llama_context::get_moe_test_frame() const {
    return moe_test_frame;
}

bool llama_context::shares_workspace_with(const llama_context & other) const {
    return sched_buffer_owner == &other || sched_buffer_borrower == &other;
}

int llama_context::attach_shared_workspace(llama_context & owner) {
    if (this != &owner && source_core_enabled() && owner.source_core_enabled() &&
            model.moe_expert_cache_slots() > 0 && owner.model.moe_expert_cache_slots() > 0) {
        std::scoped_lock lock(moe_source_mutex, owner.moe_source_mutex);
        if (moe_source_callers || owner.moe_source_callers || moe_source_closed || owner.moe_source_closed ||
                moe_source_poisoned || owner.moe_source_poisoned) { return -1; }
        if (!has_evaluated_once && !gf_res_prev_active) {
            sched_decode_outputs = std::max(sched_decode_outputs, owner.sched_decode_outputs);
        }
        return 0;
    }
    if (this == &owner ||
            !cparams.phase_aware_workspace || !owner.cparams.phase_aware_workspace ||
            model.hparams.no_alloc || owner.model.hparams.no_alloc ||
            sched_buffer_owner != nullptr || sched_buffer_borrower != nullptr ||
            owner.sched_buffer_owner != nullptr || owner.sched_buffer_borrower != nullptr) {
        return 0;
    }

    const bool compatible = std::any_of(backend_buft.begin(), backend_buft.end(), [&](ggml_backend_buffer_type_t buft) {
        return std::find(owner.backend_buft.begin(), owner.backend_buft.end(), buft) != owner.backend_buft.end();
    });
    if (!compatible) {
        return 0;
    }

    synchronize();
    owner.synchronize();
    reset_sched_workspace();

    try {
        const auto owner_plan = owner.make_sched_reserve_plan(0);
        owner.sched_reserve(owner_plan.n_tokens_max);
        owner.sched_reserve(0);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to preflight the target workspace: %s\n", __func__, err.what());
        return -1;
    }

    sched_buffer_owner = &owner;
    sched_decode_outputs = std::max(sched_decode_outputs, owner.sched_decode_outputs);

    try {
        sched_reserve(0);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to attach the draft workspace: %s\n", __func__, err.what());
        reset_sched_workspace();
        sched_buffer_owner = nullptr;
        return -1;
    }

    if (!sched_buffers_shared) {
        sched_buffer_owner = nullptr;
        return 0;
    }

    owner.sched_buffer_borrower = this;
    return 1;
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::recurrent_sparse_snapshots_supported() const {
    return memory && memory->recurrent_sparse_snapshots_supported() && recurrent_sparse_snapshot_ops_supported;
}

bool llama_context::memory_update(bool optimize, uint32_t n_tokens_req) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        if (n_tokens_req > 0) {
            sched_reserve(n_tokens_req);
        }
        if (reset_source_core_checked() != GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK) { return false; }

        // reset the previous graph results to make sure that they won't be reused
        // TODO: make mctx->apply() report if a graph reserve is needed, then reset graph results only if the memory module reset the scheduler
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        gf_res_prev_active = nullptr;

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const bool live_kv = cparams.live_context_workspace && sched_reserved_kv > 0;
        const auto mctx = live_kv ? memory->init_reserve(sched_reserved_kv) : memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = sched_reserved_tokens > 0 ? sched_reserved_tokens :
                std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (sampling.samplers.empty() || !sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (sampling.samplers.empty() || !sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (sampling.samplers.empty() || !sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (!sampling.samplers.empty() && sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (sampling.samplers.empty() || !sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (sampling.samplers.empty() || !sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (sampling.samplers.empty() || !sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //request_sched_reserve();
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    if (cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked) {
        // these flags change the graph shape
        request_sched_reserve();
    }

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    request_sched_reserve();
}

bool llama_context::set_ple_prefetch(bool enabled) {
    if (staged_inputs_checked) { return false; }
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
    auto set_callback = reinterpret_cast<ggml_backend_set_get_rows_callback_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_get_rows_callback"));
    if (!set_callback) { return false; }
    synchronize();
    set_callback(backend_cpu, enabled ? +[](const ggml_tensor * table, const ggml_tensor * indices, void * data) {
        const auto * model = static_cast<const llama_model *>(data);
        // Input setters already prefetch these tables.
        if (!model->can_prefetch.count(table)) {
            model->prefetch_rows(table, indices);
        }
    } : nullptr, enabled ? const_cast<llama_model *>(&model) : nullptr);
    ple_prefetch = enabled;
    LLAMA_LOG_INFO("%s: lazy row prefetch %s for CPU GET_ROWS\n", __func__, enabled ? "enabled" : "disabled");
    return true;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    // no scheduler reserve needed because graph shapes must not depend on causal_attn, a flip only rebuilds the graph
    //request_sched_reserve();
}

bool llama_context::get_causal_attn() const {
    return cparams.causal_attn;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //request_sched_reserve();
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    if (retire_source_sampler(seq_id) != GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK) {
        throw std::runtime_error("failed to retire source graphs before changing a sampler");
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            request_sched_reserve(true);
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        request_sched_reserve(true);

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            request_sched_reserve(true);
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    request_sched_reserve(true);

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    moe_candidate_refresh_pending = true;
    request_sched_reserve();
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    request_sched_reserve();

    return res;
}

ggml_backend_t llama_context::mtp_input_backend() {
    if (!cparams.decode_boundary_overlap || cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP ||
            model.n_devices() != 1 || cparams.pipeline_parallel || cparams.cb_eval || shared_workspace_peer()) {
        return nullptr;
    }
    if (!mtp_staging_checked) {
        mtp_staging_checked = true;
        auto * backend = backends.front().get();
        auto * device = ggml_backend_get_device(backend);
        if (device && strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "CUDA") == 0) {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(device, &props);
            if (props.caps.async && props.caps.events && ggml_backend_dev_host_buffer_type(device)) {
                mtp_staging_backend = backend;
            }
        }
    }
    return mtp_staging_backend;
}

void llama_context::place_sampled_inputs(llm_graph_result * res) {
    if (auto * backend = mtp_input_backend()) {
        for (auto * tensor : res->get_inp_mtp_tensors()) {
            ggml_backend_sched_set_tensor_backend(sched.get(), tensor, backend);
        }
    }
    if (!use_sampled_input) {
        return;
    }
    for (auto * tokens : res->get_inp_token_tensors()) {
        ggml_backend_sched_set_tensor_backend(sched.get(), tokens, sampled_input_backend);
    }
    if (!use_sampled_input_async) {
        return;
    }
    for (auto * tensor = ggml_get_first_tensor(res->get_ctx()); tensor; tensor = ggml_get_next_tensor(res->get_ctx(), tensor)) {
        if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
            ggml_backend_sched_set_tensor_backend(sched.get(), tensor, sampled_input_backend);
        }
    }
}

static size_t sampled_input_staging_size(size_t size, const ggml_tensor * tensor) {
    const size_t bytes = ggml_nbytes(tensor);
    if (bytes > SIZE_MAX - 63 || GGML_PAD(bytes, 64) > SIZE_MAX - size) {
        throw std::runtime_error("sampled input staging size overflow");
    }
    return size + GGML_PAD(bytes, 64);
}

void llama_context::set_sampled_inputs(llm_graph_result * res, const llama_ubatch & ubatch, ggml_backend_t backend, bool skip_token_upload) {
    auto & stage = sampled_staging[sampled_staging_next];
    sampled_staging_next = (sampled_staging_next + 1) % 2;
    if (stage.in_flight) {
        ggml_backend_event_synchronize(stage.uploaded.get());
        stage.in_flight = false;
    }

    struct input_storage {
        ggml_tensor * tensor;
        ggml_backend_buffer_t buffer;
        void * data;
        size_t offset;
    };
    std::vector<input_storage> inputs;
    size_t size = 0;
    const auto & token_tensors = res->get_inp_token_tensors();
    const auto append_input = [&](ggml_tensor * tensor) {
        if (!(tensor->flags & GGML_TENSOR_FLAG_INPUT) || !tensor->buffer ||
                (!skip_token_upload && ggml_backend_buffer_is_host(tensor->buffer)) ||
                (skip_token_upload && std::find(token_tensors.begin(), token_tensors.end(), tensor) != token_tensors.end())) {
            return;
        }
        GGML_ASSERT(!tensor->view_src && tensor->buffer && tensor->data);
        GGML_ASSERT(ggml_backend_sched_get_tensor_backend(sched.get(), tensor) == backend);
        inputs.push_back({tensor, tensor->buffer, tensor->data, size});
        size = sampled_input_staging_size(size, tensor);
    };
    const auto * reuse = std::getenv("GGML_MOE_INPUT_LIST_REUSE");
    if (source_core_enabled() && (!reuse || strcmp(reuse, "0"))) {
        for (auto * tensor : res->get_inp_tensors()) { append_input(tensor); }
    } else {
        for (auto * tensor = ggml_get_first_tensor(res->get_ctx()); tensor; tensor = ggml_get_next_tensor(res->get_ctx(), tensor)) {
            append_input(tensor);
        }
    }

    if (!stage.buffer || ggml_backend_buffer_get_size(stage.buffer.get()) < size) {
        auto * device = ggml_backend_get_device(backend);
        if (!cparams.decode_boundary_overlap) {
            stage.buffer.reset(ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(device), std::max<size_t>(size, 64)));
            if (!stage.buffer) {
                throw std::runtime_error("failed to allocate sampled input staging");
            }
        } else {
            const size_t previous_size = stage.buffer ? ggml_backend_buffer_get_size(stage.buffer.get()) : 64;
            const size_t capacity = std::max({size, sampled_staging_reserve, previous_size <= SIZE_MAX/2 ? 2*previous_size : previous_size});
            // Grow both slots together so the next token does not repeat the pinned allocation.
            for (auto & slot : sampled_staging) {
                if (slot.buffer && ggml_backend_buffer_get_size(slot.buffer.get()) >= capacity) {
                    continue;
                }
                if (slot.in_flight) {
                    ggml_backend_event_synchronize(slot.uploaded.get());
                    slot.in_flight = false;
                }
                auto buffer = ggml_backend_buffer_ptr(ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(device), capacity));
                if (!buffer) {
                    throw std::runtime_error("failed to allocate sampled input staging");
                }
                slot.buffer = std::move(buffer);
            }
        }
    }
    if (!stage.uploaded) {
        stage.uploaded.reset(ggml_backend_event_new(ggml_backend_get_device(backend)));
        if (!stage.uploaded) {
            throw std::runtime_error("failed to allocate sampled input event");
        }
    }

    // Input setters write into pinned storage. Restore graph pointers before submitting any work.
    auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(stage.buffer.get()));
    for (auto & input : inputs) {
        input.tensor->buffer = stage.buffer.get();
        input.tensor->data = base + input.offset;
    }
    const auto restore = [&]() {
        for (auto & input : inputs) {
            input.tensor->buffer = input.buffer;
            input.tensor->data = input.data;
        }
    };
    try {
        res->set_inputs(&ubatch, skip_token_upload);
    } catch (...) {
        restore();
        throw;
    }
    restore();
    for (const auto & input : inputs) {
        ggml_backend_tensor_set_async(backend, input.tensor, base + input.offset, 0, ggml_nbytes(input.tensor));
    }
    ggml_backend_event_record(stage.uploaded.get(), backend);
    stage.in_flight = true;
}

llm_graph_result * llama_context::process_ubatch(
        const llama_ubatch & ubatch,
        llm_graph_type gtype,
        llama_memory_context_i * mctx,
        ggml_status & ret,
        const llama_graph_execution_intent * execution_intent) {
    mtp_draft_vocab_locked = true;
    if (moe_source_poisoned || moe_source_closed) { ret = GGML_STATUS_FAILED; return nullptr; }
    refresh_moe_layer_owners();
    if (moe_source_poisoned || moe_source_closed) { ret = GGML_STATUS_FAILED; return nullptr; }
    const bool hybrid_speculative = moe_hybrid_required && execution_intent != nullptr &&
        execution_intent->domain == GGML_GRAPH_EXECUTION_DOMAIN_MAIN &&
        execution_intent->row_semantics == GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SPECULATIVE;
    const bool hybrid_independent = moe_hybrid_required && source_core_enabled() && ubatch.n_tokens > 1 &&
        !hybrid_speculative && ubatch_has_independent_rows(ubatch);
    const bool hybrid_auxiliary = moe_hybrid_required && source_core_enabled() && execution_intent != nullptr &&
        execution_intent->domain != GGML_GRAPH_EXECUTION_DOMAIN_MAIN;
    const bool hybrid_prefill = moe_hybrid_required && source_core_enabled() &&
        cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT && execution_intent == nullptr && ubatch.n_tokens > 1 && ubatch_has_sequential_spans(ubatch);
    if (moe_hybrid_required && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT && ubatch.n_tokens > 1 &&
            !hybrid_speculative && !hybrid_independent && !hybrid_auxiliary && !hybrid_prefill) {
        LLAMA_LOG_ERROR("%s: hybrid prompt must use the source prefill pipeline\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    const uint32_t hybrid_rows = moe_hybrid_required && (ubatch.n_tokens == 1 || hybrid_speculative || hybrid_independent || hybrid_auxiliary || hybrid_prefill) ? ubatch.n_tokens : 0;
    ggml_graph_execution_certificate requested_certificate = {};
    if (moe_hybrid_required && !make_graph_execution_certificate(
            &ubatch, execution_intent, moe_required_grouped_execution_supported, requested_certificate)) {
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    const bool source_capacity = moe_source_graph_capacity && source_core_enabled() && hybrid_rows != 0 && !hybrid_prefill &&
        std::any_of(model.moe_sources().begin(), model.moe_sources().end(), [](const llama_moe_source_group & source) {
            return std::any_of(source.banks.begin(), source.banks.end(), [](const llama_moe_source_bank & bank) {
                return bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE && is_moe_cached_tensor(bank.tensor);
            });
        });
    if (mctx && !mctx->apply(source_capacity)) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    const bool source_cache = source_graph_cache_enabled();
    sched_reserve_plan source_plan;
    if (source_cache) {
        acquire_shared_workspace();
        source_plan = make_sched_reserve_plan(ubatch.n_tokens,
            cparams.live_context_workspace && mctx ? mctx->get_attn_reserve_n_kv() : 0);
        prepare_sched_reserve(source_plan);
        if (!ggml_backend_sched_refresh_resizable_plan(sched.get())) { ret = GGML_STATUS_ALLOC_FAILED; return nullptr; }
    }
    if (!select_source_graph_variant(hybrid_rows ? requested_certificate : ggml_graph_execution_certificate{})) { ret = GGML_STATUS_FAILED; return nullptr; }
    if (source_cache) {
        sched_reserved_tokens = source_plan.n_tokens;
        sched_reserved_kv = source_plan.live_kv ? source_plan.n_kv : 0;
        sched_need_reserve = false;
        sched_sampler_reserve_only = false;
    }

    auto * res = get_gf_res_prev();
    auto * gf  = res->get_gf();

    // Graph reuse must include the full topology and input compatibility.
    const auto gparams = graph_params(res, ubatch, mctx, gtype);
    if (moe_hybrid_required && memcmp(&moe_hybrid_graph_certificate, &requested_certificate, sizeof(requested_certificate))) {
        gf_res_prev_active = nullptr;
    }

    bool reactivated_mtp = false;
    if (moe_source_graph_capacity && cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
            !graph_reuse_disable && !cparams.pipeline_parallel && !cparams.cb_eval &&
            !use_sampled_input_async && !sampled_inputs_device && gf_res_prev_active != res &&
            ggml_graph_n_nodes(gf) > 0 && !ggml_backend_sched_moe_source_selected_v1(sched.get())) {
        uint64_t generation = 0;
        uint64_t shrink_generation = 0;
        ggml_backend_sched_get_buffer_state(sched.get(), &generation, &shrink_generation);
        const size_t index = n_outputs > 0;
        if (generation != 0 && mtp_graph_buffer_generation[index] == generation &&
                mtp_graph_shrink_generation[index] == shrink_generation && res->can_reuse(gparams)) {
            acquire_shared_workspace();
            ggml_backend_sched_synchronize(sched.get());
            workspace_in_flight = false;
            gf_res_prev_active = nullptr;
            ggml_backend_sched_reset(sched.get());
            // Scheduler copies belong to its active graph. Restore the original inputs before splitting again.
            for (const auto & binding : mtp_graph_original_sources[index]) {
                std::copy(binding.sources.begin(), binding.sources.end(), binding.tensor->src);
            }
            // Reallocate scratch tensors; model and memory buffers retain their owners.
            for (auto * tensor = ggml_get_first_tensor(res->get_ctx()); tensor;
                    tensor = ggml_get_next_tensor(res->get_ctx(), tensor)) {
                if (tensor->buffer && ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
                    tensor->buffer = nullptr;
                    tensor->data = nullptr;
                    tensor->extra = nullptr;
                }
            }
            ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
            place_moe_regions(res);
            place_sampled_inputs(res);
            if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
                ret = GGML_STATUS_ALLOC_FAILED;
                return nullptr;
            }
            if (!finalize_moe_regions(res, false, nullptr)) {
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
            prepare_required_grouped_execution(res);
            ggml_backend_sched_get_buffer_state(sched.get(), &mtp_graph_buffer_generation[index],
                    &mtp_graph_shrink_generation[index]);
            gf_res_prev_active = res;
            reactivated_mtp = true;
            if (std::getenv("GGML_MOE_GRAPH_REUSE_DIAGNOSTIC")) {
                fprintf(stderr, "moe-graph-reuse: reactivate_mtp rows=%u outputs=%d\n", ubatch.n_tokens, n_outputs);
            }
        }
    }

    if (reactivated_mtp || (!graph_reuse_disable && gf_res_prev_active == res && sampled_inputs_device == use_sampled_input_async && res->can_reuse(gparams))) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
            workspace_in_flight = false;
        }

        n_reused++;
    } else {
        if (std::getenv("GGML_MOE_GRAPH_REUSE_DIAGNOSTIC")) {
            fprintf(stderr, "moe-graph-reuse: rebuild rows=%u active=%u sampled_match=%u disabled=%u nodes=%d\n",
                ubatch.n_tokens, unsigned(gf_res_prev_active == res), unsigned(sampled_inputs_device == use_sampled_input_async), unsigned(graph_reuse_disable), ggml_graph_n_nodes(gf));
        }
        gf_res_prev_active = nullptr;
        bool rebuild_async = cparams.decode_boundary_overlap && use_sampled_input_async && sampled_inputs_device && !cparams.cb_eval && !cparams.pipeline_parallel;
        for (int i = 0; rebuild_async && i < ggml_graph_n_nodes(gf); ++i) {
            auto * node = ggml_graph_node(gf, i);
            if (node->op != GGML_OP_NONE && !ggml_is_view(node) &&
                    ggml_backend_sched_get_tensor_backend(sched.get(), node) != sampled_input_backend) {
                rebuild_async = false;
            }
        }
        // CUDA consumes graph metadata at submission. Reallocation still fences the backing buffers in the scheduler.
        if ((use_sampled_input_async || sampled_inputs_device) && !rebuild_async) {
            ggml_backend_sched_synchronize(sched.get());
            workspace_in_flight = false;
        }
        if (ggml_backend_sched_moe_source_selected_v1(sched.get())) {
            if (reset_source_core_checked(false) != GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK) { ret = GGML_STATUS_FAILED; return nullptr; }
            res->reset();
        } else {
            res->reset();
            ggml_backend_sched_reset(sched.get());
        }
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (moe_hybrid_metadata && !res->discover_moe_regions(model.moe_sources(), is_moe_cached_tensor)) {
            LLAMA_LOG_ERROR("%s: incomplete canonical MoE projection coverage\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (moe_source_graph_capacity && cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
            if (gtype == LLM_GRAPH_TYPE_DECODER_MTP && n_outputs == 0 && !gparams.is_reserve && !cparams.cb_eval) {
                res->retain_state_computation();
            }
            auto & bindings = mtp_graph_original_sources[n_outputs > 0];
            bindings.clear();
            bindings.reserve(ggml_graph_n_nodes(gf));
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                mtp_graph_sources binding = {};
                binding.tensor = ggml_graph_node(gf, i);
                std::copy(std::begin(binding.tensor->src), std::end(binding.tensor->src), binding.sources.begin());
                bindings.push_back(binding);
            }
        }
        place_moe_regions(res);
        place_sampled_inputs(res);
        sampled_inputs_device = use_sampled_input_async;
        moe_source_active_input_mode = use_sampled_input_async ? 2 : unsigned(use_sampled_input);
        const bool allocated = rebuild_async ? ggml_backend_sched_alloc_graph_async(sched.get(), gf) :
                                               ggml_backend_sched_alloc_graph(sched.get(), gf);
        if (!allocated) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
        if (source_core_enabled()) {
            ggml_backend_sched_get_buffer_state(sched.get(), &sched_buffer_generation, &sched_shrink_generation);
            sched_source_retirement_epoch = ggml_backend_sched_moe_source_retirement_epoch_v1(sched.get());
        }
        ggml_graph_execution_certificate hybrid_certificate = {};
        if (moe_hybrid_required && (!make_graph_execution_certificate(
                &ubatch, execution_intent, moe_graph_supports_required_grouped(gf), hybrid_certificate) ||
                (hybrid_rows != 0 && hybrid_certificate.magic != GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC))) {
            LLAMA_LOG_ERROR("%s: hybrid preparation has no validated execution certificate\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
        if (!finalize_moe_regions(res, hybrid_rows != 0, hybrid_rows != 0 ? &hybrid_certificate : nullptr)) {
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
        prepare_required_grouped_execution(res);
        moe_hybrid_graph_certificate = hybrid_certificate;
        moe_source_active_outputs = n_outputs;

        if (use_sampled_input && !res->can_decode_sampled()) {
            LLAMA_LOG_ERROR("%s: rebuilt graph requires host token inputs\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
        if (use_sampled_input_async) {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                auto * node = ggml_graph_node(gf, i);
                if (node->op != GGML_OP_NONE && !ggml_is_view(node) &&
                        ggml_backend_sched_get_tensor_backend(sched.get(), node) != sampled_input_backend) {
                    LLAMA_LOG_ERROR("%s: rebuilt graph requires another backend for %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                    ret = GGML_STATUS_FAILED;
                    return nullptr;
                }
            }
        }
        gf_res_prev_active = res;
        if (moe_source_graph_capacity && cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
            const size_t index = n_outputs > 0;
            ggml_backend_sched_get_buffer_state(sched.get(), &mtp_graph_buffer_generation[index],
                    &mtp_graph_shrink_generation[index]);
        }
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        acquire_shared_workspace();
        if (use_sampled_input) {
            if (use_sampled_input_async) {
                set_sampled_inputs(res, ubatch, sampled_input_backend);
            } else {
                res->set_inputs(&ubatch, true);
            }
            for (auto * tokens : res->get_inp_token_tensors()) {
                if (!tokens->buffer) {
                    continue;
                }
                GGML_ASSERT(ggml_backend_sched_get_tensor_backend(sched.get(), tokens) == sampled_input_backend);
                GGML_ASSERT(ggml_nelements(tokens) == ubatch.n_tokens);
                for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                    GGML_ASSERT(ubatch.n_seq_id[i] == 1);
                    auto * source = sampled_input_by_seq.at(ubatch.seq_id[i][0]);
                    GGML_ASSERT(source);
                    ggml_tensor row = *source;
                    row.buffer = tokens->buffer;
                    row.data = static_cast<char *>(tokens->data) + i*sizeof(llama_token);
                    row.view_src = tokens;
                    row.view_offs = i*sizeof(llama_token);
                    ggml_backend_tensor_copy_async(sampled_input_backend, sampled_input_backend, source, &row);
                }
            }
        } else {
            auto * backend = mtp_input_backend();
            bool stage_mtp = backend != nullptr;
            bool has_device_inputs = false;
            for (auto * tensor = ggml_get_first_tensor(res->get_ctx()); stage_mtp && tensor; tensor = ggml_get_next_tensor(res->get_ctx(), tensor)) {
                if (!(tensor->flags & GGML_TENSOR_FLAG_INPUT) || !tensor->buffer || ggml_backend_buffer_is_host(tensor->buffer)) {
                    continue;
                }
                stage_mtp = !tensor->view_src && tensor->data && ggml_backend_sched_get_tensor_backend(sched.get(), tensor) == backend;
                has_device_inputs = true;
            }
            if (stage_mtp && has_device_inputs) {
                set_sampled_inputs(res, ubatch, backend, false);
            } else {
                res->set_inputs(&ubatch);
            }
        }

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    if (moe_test_hook_set && hybrid_rows &&
            !ggml_backend_sched_moe_hybrid_set_test_hook_v1(sched.get(), moe_test_hook, moe_test_hook_data)) {
        LLAMA_LOG_ERROR("%s: failed to bind context MoE observer to prepared sessions\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1, &ubatch, execution_intent);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    sampled_output_positions.clear();
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.output[i]) {
            sampled_output_positions.emplace_back(ubatch.n_seq_id[i] == 1 ? ubatch.seq_id[i][0] : -1, ubatch.pos[i]);
        }
    }
    ret = GGML_STATUS_SUCCESS;

    return res;
}

int32_t llama_context::decode_sampled(llama_seq_id seq_id, llama_pos pos, llama_token * previous) {
    const llama_sampled_decode_item item = {seq_id, pos};
    return decode_sampled(&item, 1, previous);
}

int32_t llama_context::decode_sampled(const llama_sampled_decode_item * items, int32_t n_items, llama_token * previous) {
    llama_source_call_guard source_guard(*this);
    if (!source_guard.entered) { return -3; }
    if (!items || n_items <= 0 || (uint32_t) n_items > n_seq_max() ||
            (uint32_t) n_items > cparams.n_batch || (uint32_t) n_items > cparams.n_ubatch ||
            model.n_devices() != 1 || cparams.pipeline_parallel || cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT ||
            cparams.embeddings || cparams.embeddings_nextn || cparams.cb_eval || !memory || n_outputs == 0) {
        return 1;
    }

    auto * res = gf_res_prev_active;
    if (!res) {
        return 1;
    }
    const bool host_inputs = !res->can_decode_sampled() || !memory->can_decode_sampled();
    if ((host_inputs && (!previous || !can_decode_sampled_host())) || sampled_output_positions.size() != res->t_sampled.size()) {
        return 1;
    }

    std::vector<ggml_tensor *> sources;
    std::vector<bool> seen(n_seq_max(), false);
    for (int32_t i = 0; i < n_items; ++i) {
        const auto & item = items[i];
        if (item.seq_id < 0 || (uint32_t) item.seq_id >= n_seq_max() || seen[item.seq_id] ||
                sampling.samplers.count(item.seq_id) != 1 || item.pos < 0 || item.pos != memory->seq_pos_max(item.seq_id) + 1) {
            return 1;
        }
        seen[item.seq_id] = true;
        auto * chain = sampling.samplers.at(item.seq_id);
        const int n_samplers = llama_sampler_chain_n(chain);
        if (n_samplers < 1) {
            return 1;
        }
        const char * terminal = llama_sampler_name(llama_sampler_chain_get(chain, n_samplers - 1));
        if (strcmp(terminal, "+greedy") != 0 &&
                (strcmp(terminal, "+dist") != 0 || cparams.n_outputs_max_per_seq != 1)) {
            return 1;
        }
        for (int j = 0; j + 1 < n_samplers; ++j) {
            const std::string name = llama_sampler_name(llama_sampler_chain_get(chain, j));
            if (name != "+logit-bias" && name != "+top-k" && name != "+top-p" && name != "+min-p" &&
                    name != "+temp" && name != "+temp-ext" && name != "?top-k" && name != "?top-p" &&
                    name != "?min-p" && name != "?temp" && name != "?temp-ext" && name != "?typical" &&
                    name != "?xtc" && name != "?penalties" && name != "?dry" && name != "?top-n-sigma" && name != "?logit-bias") {
                return 1;
            }
        }
        const auto position = std::make_pair(item.seq_id, item.pos - 1);
        const auto it = std::find(sampled_output_positions.begin(), sampled_output_positions.end(), position);
        if (it == sampled_output_positions.end()) {
            return 1;
        }
        auto * source = res->t_sampled[it - sampled_output_positions.begin()];
        if (!source || source->type != GGML_TYPE_I32 || ggml_nelements(source) != 1) {
            return 1;
        }
        sources.push_back(source);
    }

    auto * backend = ggml_backend_sched_get_tensor_backend(sched.get(), sources[0]);
    auto * device = backend ? ggml_backend_get_device(backend) : nullptr;
    if (!device || strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "CUDA") != 0 ||
            (!host_inputs && (!model.tok_embd || !model.tok_embd->buffer || ggml_backend_buffer_is_host(model.tok_embd->buffer) ||
             ggml_backend_buft_get_device(ggml_backend_buffer_get_type(model.tok_embd->buffer)) != device))) {
        return 1;
    }
    for (auto * source : sources) {
        if (ggml_backend_sched_get_tensor_backend(sched.get(), source) != backend || ggml_backend_buffer_is_host(source->buffer)) {
            return 1;
        }
    }
    if (host_inputs) {
        if (!loras->empty()) {
            return decode_sampled_host(items, n_items, sources, previous);
        }
        if (!staged_inputs_checked) {
            staged_inputs = llama_staged_inputs::create(model, backend, ple_prefetch);
            staged_inputs_checked = true;
            LLAMA_LOG_INFO("%s: Flash Next staged inputs %s\n", __func__, staged_inputs ? "enabled" : "unavailable, using host inputs");
            if (staged_inputs && ple_prefetch) {
                LLAMA_LOG_INFO("%s: staged PLE prefetch enabled\n", __func__);
            }
        }
        if (!staged_inputs) {
            return decode_sampled_host(items, n_items, sources, previous);
        }
    }
    const auto & token_tensors = res->get_inp_token_tensors();
    const bool move_tokens = std::any_of(token_tensors.begin(), token_tensors.end(), [&](ggml_tensor * input) {
        return ggml_backend_sched_get_tensor_backend(sched.get(), input) != backend;
    });
    if (previous) {
        if (shared_workspace_peer() || !ggml_backend_dev_host_buffer_type(device)) {
            return 1;
        }
        auto * gf = res->get_gf();
        for (int i = 0; !host_inputs && i < ggml_graph_n_nodes(gf); ++i) {
            auto * node = ggml_graph_node(gf, i);
            if (node->op != GGML_OP_NONE && !ggml_is_view(node) &&
                    ggml_backend_sched_get_tensor_backend(sched.get(), node) != backend) {
                return 1;
            }
        }
    }

    if (!sampled_input) {
        sampled_input_ctx.reset(ggml_init({ggml_tensor_overhead()*(n_seq_max() + 1), nullptr, true}));
        if (!sampled_input_ctx) {
            return -2;
        }
        auto * input = ggml_new_tensor_1d(sampled_input_ctx.get(), GGML_TYPE_I32, n_seq_max());
        ggml_set_name(input, "decode_sampled_input");
        sampled_input_rows.clear();
        for (uint32_t i = 0; i < n_seq_max(); ++i) {
            sampled_input_rows.push_back(ggml_view_1d(sampled_input_ctx.get(), input, 1, i*sizeof(llama_token)));
        }
        sampled_input_buf.reset(ggml_backend_alloc_ctx_tensors(sampled_input_ctx.get(), backend));
        if (!sampled_input_buf) {
            return -2;
        }
        sampled_input = input;
        sampled_input_backend = backend;
    }
    if (sampled_input_backend != backend) {
        return 1;
    }
    if (!sampled_output_ready || !sampled_output_host) {
        sampled_output_ready.reset(ggml_backend_event_new(device));
        sampled_output_host.reset(ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(device), n_seq_max()*sizeof(llama_token)));
        if (!sampled_output_ready || !sampled_output_host) {
            return -2;
        }
    }

    // Preserve every selected output before decode can reuse the graph and host storage.
    sampled_input_by_seq.assign(n_seq_max(), nullptr);
    for (int32_t i = 0; i < n_items; ++i) {
        ggml_backend_tensor_copy_async(backend, backend, sources[i], sampled_input_rows[i]);
        sampled_input_by_seq[items[i].seq_id] = sampled_input_rows[i];
    }
    auto * host = static_cast<llama_token *>(ggml_backend_buffer_get_base(sampled_output_host.get()));
    ggml_backend_tensor_get_async(backend, sampled_input, host, 0, n_items*sizeof(llama_token));
    ggml_backend_event_record(sampled_output_ready.get(), backend);
    if (host_inputs) {
        staged_inputs->set_source(host, sampled_output_ready.get());
    }

    std::vector<llama_token> tokens(n_items, 0);
    std::vector<llama_pos> positions(n_items);
    std::vector<llama_seq_id> seq_ids(n_items);
    std::vector<llama_seq_id *> seq_ptrs(n_items);
    std::vector<int32_t> n_seq_ids(n_items, 1);
    std::vector<int8_t> outputs(n_items, 1);
    for (int32_t i = 0; i < n_items; ++i) {
        positions[i] = items[i].pos;
        seq_ids[i] = items[i].seq_id;
        seq_ptrs[i] = &seq_ids[i];
    }
    if (!previous) {
        ggml_backend_event_synchronize(sampled_output_ready.get());
        std::copy_n(host, n_items, tokens.data());
    }
    llama_batch batch = {n_items, tokens.data(), nullptr, positions.data(), n_seq_ids.data(), seq_ptrs.data(), outputs.data()};
    use_sampled_input = true;
    use_sampled_input_async = previous != nullptr;
    if (!source_graph_cache_enabled() && (move_tokens || (use_sampled_input_async && !sampled_inputs_device))) {
        request_sched_reserve();
    }
    const int64_t preceding_tokens = n_queued_tokens;
    const uint64_t preceding_generation = compute_sync_generation;
    int32_t ret;
    try {
        ret = decode(batch);
        if (host_inputs) {
            staged_inputs->finish();
        }
        if (previous) {
            ggml_backend_event_synchronize(sampled_output_ready.get());
            if (preceding_generation == compute_sync_generation && preceding_tokens > 0) {
                const int64_t now = ggml_time_us();
                finish_compute(preceding_tokens, now - t_compute_start_us);
                n_queued_tokens -= preceding_tokens;
                t_compute_start_us = n_queued_tokens > 0 ? now : 0;
            }
            std::copy_n(host, n_items, previous);
            if (ret == 0) {
                for (int32_t i = 0; i < n_items; ++i) {
                    memory->seq_set_last_token(items[i].seq_id, items[i].pos, previous[i]);
                }
            }
        }
    } catch (...) {
        use_sampled_input = false;
        use_sampled_input_async = false;
        if (host_inputs && source_core_enabled()) {
            // An input failure can follow partial graph effects.
            moe_source_poisoned.store(true);
        }
        if (host_inputs) {
            try { staged_inputs->finish(); } catch (...) {}
        }
        synchronize();
        throw;
    }
    use_sampled_input = false;
    use_sampled_input_async = false;
    return ret == 0 ? 0 : (ret < 0 ? ret : -3);
}

int llama_context::encode(const llama_batch_ext & batch_inp) {
    llama_source_call_guard source_guard(*this);
    if (!source_guard.entered) { return -3; }
    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != hparams.n_embd_inp_enc()) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the encoder input %u\n",
                __func__, batch_inp.n_embd, hparams.n_embd_inp_enc());
        return -1;
    }

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always pass the full sequence starting from pos = 0
    llama_graph_execution_intent execution_intent;
    llama_speculative_execution_policy speculative_policy;
    llama_speculative_execution_policy * speculative_policy_out =
        cparams.ctx_type == LLAMA_CONTEXT_TYPE_DRAFT ? &speculative_policy : nullptr;
    if (!batch_init_with_speculative_execution_policy(
            *balloc, cparams.ctx_type, batch_inp, model.vocab, true,
            model.moe_expert_cache_slots(), moe_required_grouped_execution_supported, speculative_policy_out)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    if (speculative_policy_out != nullptr) {
        if (speculative_policy.fail_closed) {
            LLAMA_LOG_ERROR("%s: unsupported speculative grouped MoE batch structure\n", __func__);
            return -2;
        }
        if (speculative_policy.preserve_intent) {
            execution_intent.domain = speculative_policy.domain;
            execution_intent.row_semantics = speculative_policy.row_semantics;
        }
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    try {
        if (!source_graph_cache_enabled() || (sched_need_reserve && !sched_sampler_reserve_only)) { sched_reserve(n_tokens); }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to reserve compute workspace: %s\n", __func__, err.what());
        return -2;
    }
    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(
        ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status,
        speculative_policy.preserve_intent ? &execution_intent : nullptr);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch_ext & batch_inp, const llama_decode_execution_intent * intent) {
    llama_source_call_guard source_guard(*this);
    if (!source_guard.entered) { return -3; }
    if (moe_source_poisoned || moe_source_closed) { return -3; }
    llama_graph_execution_intent execution_intent;
    bool has_execution_intent = intent != nullptr;
    if (intent && cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT) { return -1; }

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != batch_inp.n_embd_inp) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the decoder input %zu\n",
                __func__, batch_inp.n_embd, batch_inp.n_embd_inp);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    // embedding contexts output every token even when no token is explicitly marked as output
    if (has_samplers) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (const auto & tok : batch_inp.tokens) {
            if (!output_all && !tok.output) {
                continue;
            }

            for (auto seq_id : tok.seq_ids) {
                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                auto sampler = sampling.samplers.find(seq_id);
                if (sampler != sampling.samplers.end() &&
                        seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq) {
                    LLAMA_LOG_ERROR("%s: backend sampling supports at most %u outputs per sequence "
                            "(seq_id %d had %d)\n", __func__, cparams.n_outputs_max_per_seq,
                            seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    llama_speculative_execution_policy speculative_policy;
    llama_speculative_execution_policy * speculative_policy_out = !has_execution_intent &&
        (cparams.ctx_type == LLAMA_CONTEXT_TYPE_DRAFT || cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) ?
            &speculative_policy : nullptr;
    if (!batch_init_with_speculative_execution_policy(
            *balloc, cparams.ctx_type, batch_inp, vocab, output_all,
            model.moe_expert_cache_slots(), moe_required_grouped_execution_supported, speculative_policy_out)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    if (speculative_policy_out != nullptr) {
        if (speculative_policy.fail_closed) {
            LLAMA_LOG_ERROR("%s: unsupported speculative grouped MoE batch structure\n", __func__);
            return -2;
        }
        if (speculative_policy.preserve_intent) {
            execution_intent.domain = speculative_policy.domain;
            execution_intent.row_semantics = speculative_policy.row_semantics;
            has_execution_intent = true;
        }
    }

    if (intent && !target_verification_intent_valid(intent, balloc->get_batch(), execution_intent)) {
        LLAMA_LOG_ERROR("%s: invalid target verification execution intent\n", __func__);
        return -1;
    }

    if (!moe_hybrid_execution_supported(moe_hybrid_required, execution_intent.domain, execution_intent.row_semantics)) {
        LLAMA_LOG_ERROR("%s: required MAIN speculative hybrid execution is not supported\n", __func__);
        return -3;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    const bool live_exact_batch_plan = cparams.live_context_workspace;

    output_swaps.clear();
    embd_batch_idxs.clear();

    if (!live_exact_batch_plan) {
        try {
            if (!source_graph_cache_enabled() || (sched_need_reserve && !sched_sampler_reserve_only)) { sched_reserve(n_tokens_all); }
        } catch (const std::exception & err) {
            LLAMA_LOG_ERROR("%s: failed to reserve compute workspace: %s\n", __func__, err.what());
            return -2;
        }
    }

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    bool did_optimize = false;

    // handle any pending shifts/copies
    try {
        memory_update(false, live_exact_batch_plan ? n_tokens_all : 0);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to update memory: %s\n", __func__, err.what());
        return -2;
    }

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        bool optimized;
                        try {
                            optimized = memory_update(true, live_exact_batch_plan ? n_tokens_all : 0);
                        } catch (const std::exception & err) {
                            LLAMA_LOG_ERROR("%s: failed to optimize memory: %s\n", __func__, err.what());
                            return -2;
                        }
                        if (optimized) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    if (live_exact_batch_plan) {
        try {
            if (!source_graph_cache_enabled() || (sched_need_reserve && !sched_sampler_reserve_only)) { sched_reserve(n_tokens_all, mctx->get_attn_reserve_n_kv()); }
        } catch (const std::exception & err) {
            LLAMA_LOG_ERROR("%s: failed to reserve live-context workspace: %s\n", __func__, err.what());
            return -2;
        }
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    do {
        const auto & ubatch = mctx->get_ubatch();
        if (has_execution_intent && !ubatch_matches_graph_execution_intent(
                cparams.ctx_type, model.moe_expert_cache_slots(), ubatch, execution_intent)) {
            LLAMA_LOG_ERROR("%s: ubatch does not match the validated execution intent\n", __func__);
            return -2;
        }

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        const auto * res = process_ubatch(
            ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status,
            has_execution_intent ? &execution_intent : nullptr);

        if (!res) {
            if (moe_hybrid_required) {
                // A failed hybrid turn can have written KV or recurrent state.
                if (ggml_backend_sched_moe_source_selected_v1(sched.get())) {
                    if (reset_source_core_checked() != GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK) {
                        LLAMA_LOG_ERROR("%s: source core drain failed; graph and memory owners retained\n", __func__);
                        return -3;
                    }
                } else {
                    ggml_backend_sched_synchronize(sched.get());
                    ggml_backend_sched_reset(sched.get());
                }
                memory->clear(true);
                for (auto & previous : gf_res_prev) { if (previous) { previous->reset(); } }
                gf_res_prev_active = nullptr;
                if (gf_res_reserve) { gf_res_reserve->reset(); }
                moe_hybrid_graph_certificate = {};
                n_outputs = 0;
                ++graph_execution_owner_generation;
                request_sched_reserve();
                LLAMA_LOG_ERROR("%s: failed required target window drained; target memory and graph state invalidated\n", __func__);
                return -3;
            }
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        bool extract_all_idxs = extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
                extract_all_idxs = extract_all_idxs || !masked;
            }
        }

        if (extract_all_idxs) {
            GGML_ASSERT(ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens);
            GGML_ASSERT(embd_batch_idxs.size() == (size_t) n_tokens_prev);
            const auto & batch_idxs = ubatch.data->batch_idxs;
            embd_batch_idxs.insert(embd_batch_idxs.end(), batch_idxs.begin(), batch_idxs.end());
        }

        if (has_samplers) {
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
            copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched.get(), &sampling.logits_count);
            copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched.get(), &sampling.probs_count);
            copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched.get(), &sampling.candidates_count);
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = true;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        if (!use_sampled_input_async) {
            std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
        }
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

bool llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    bool extracted = false;
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        GGML_ASSERT(row_floats == model.hparams.n_embd);
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        // Tensor-split backends require a zero source offset.
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
        extracted = true;
    }
    return extracted;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab     = model.vocab.n_tokens();
    const uint64_t n_embd      = model.hparams.n_embd;
    const uint64_t n_embd_out  = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0*n_embd_out + k], embd.data[i1*n_embd_out + k]);
            }
        }

        if (embd_nextn.size > 0 && cparams.embeddings_nextn_masked) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0*n_embd_out + k], embd_nextn.data[i1*n_embd_out + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();

    // [TAG_EXTRACT_TARGET_EMBEDDINGS]
    // Layer inputs and unmasked NextN embeddings contain all token rows, independent of logits selection.
    for (size_t i = 0; i < embd_batch_idxs.size(); ++i) {
        while (embd_batch_idxs[i] != (int32_t) i) {
            const int32_t j = embd_batch_idxs[i];
            GGML_ASSERT(j >= 0 && (size_t) j < embd_batch_idxs.size());
            if (embd_nextn.has_data() && !cparams.embeddings_nextn_masked) {
                for (size_t k = 0; k < n_embd_out; ++k) {
                    std::swap(embd_nextn.data[i*n_embd_out + k], embd_nextn.data[j*n_embd_out + k]);
                }
            }
            for (auto & layer : embd_layer_inp) {
                if (layer.has_data()) {
                    for (size_t k = 0; k < n_embd; ++k) {
                        std::swap(layer.data[i*n_embd + k], layer.data[j*n_embd + k]);
                    }
                }
            }
            std::swap(embd_batch_idxs[i], embd_batch_idxs[j]);
        }
    }
    embd_batch_idxs.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    uint32_t res;
    if (model.arch == LLM_ARCH_KIMI_K3 || model.arch == LLM_ARCH_GLM5_NEXT) {
        // the n_tokens*40 budget below is exhausted at ubatch 3840
        res = std::max<uint32_t>(n_tokens * 160, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_HRM_TEXT) {
        // the 128-slot looped graph needs roughly one stack per token budget
        res = std::max<uint32_t>(n_tokens * 80, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_BAILINGMOE3 ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_CLEF ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0) ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_01 ||
        model.arch == LLM_ARCH_MINIMAX_M3 ||
        model.arch == LLM_ARCH_HY_V4) {
        res = std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
        res = std::max<uint32_t>(1024u, 12u*model.n_tensors());
    } else {
        res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }

    uint32_t n_sampling_nodes = 0;
    uint32_t n_sampling_nodes_max = 0;
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        const uint32_t n_nodes = llama_sampler_backend_n_nodes(sampler);
        n_sampling_nodes += n_nodes;
        if (cparams.n_outputs_max_per_seq > 1) {
            n_sampling_nodes_max = std::max(n_sampling_nodes_max, n_nodes);
        }
    }

    const uint32_t n_sampling_outputs_max = std::min<uint64_t>(
            std::min(n_tokens, cparams.n_outputs_max),
            (uint64_t) cparams.n_seq_max * cparams.n_outputs_max_per_seq);

    res += n_sampling_nodes;
    if (n_sampling_outputs_max > 1) {
        res += (n_sampling_outputs_max - 1) * n_sampling_nodes_max;
    }

    if (cparams.training) {
        res *= 4;
    }

    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

llm_graph_result * llama_context::get_gf_res_prev() {
    auto & res = gf_res_prev[n_outputs > 0];
    const auto max_nodes = source_graph_cache_enabled() ?
        std::max<int64_t>(gf_res_reserve->get_max_nodes(), graph_max_nodes(std::min(cparams.n_ctx, cparams.n_ubatch))) :
        gf_res_reserve->get_max_nodes();
    if (res && res->get_max_nodes() < max_nodes) {
        GGML_ASSERT(reset_source_core_checked(false) == GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK);
        ggml_backend_sched_synchronize(sched.get());
        workspace_in_flight = false;
        gf_res_prev_active = nullptr;
        res.reset();
    }
    if (!res) {
        res.reset(new llm_graph_result(max_nodes));
    }
    return res.get();
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const llama_seq_id seq_id = entry.first;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seqs) {
            continue;
        }

        sampler_seqs.push_back(seq_id);
        has_sampler[seq_id] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    refresh_moe_layer_owners();
    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse old graphs, so we reset the previous graph results
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_prev_active = nullptr;

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    ubatch_prepare_reserve(ubatch, n_outputs, sampling.samplers, cparams.n_outputs_max_per_seq);

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type), true);

    res->reset();

    auto * gf = model.build_graph(gparams);

    if (moe_hybrid_metadata && !res->discover_moe_regions(model.moe_sources(), is_moe_cached_tensor)) {
        LLAMA_LOG_ERROR("%s: incomplete canonical MoE projection coverage\n", __func__);
        return nullptr;
    }

    if (source_core_enabled()) {
        uint32_t regions = 0, prepared = 0;
        if (!measure_moe_regions(*res, true, regions, prepared)) { return nullptr; }
        moe_source_region_capacity = std::max(moe_source_region_capacity, regions);
        moe_source_prepared_capacity = std::max(moe_source_prepared_capacity, prepared);
    }

    if (cparams.decode_boundary_overlap && n_tokens == n_seqs) {
        size_t size = 0;
        for (auto * tensor = ggml_get_first_tensor(res->get_ctx()); tensor; tensor = ggml_get_next_tensor(res->get_ctx(), tensor)) {
            if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
                size = sampled_input_staging_size(size, tensor);
            }
        }
        sampled_staging_reserve = std::max(sampled_staging_reserve, size);
    }

    this->n_input_tensors = llama_graph_n_input_tensors(gf);
    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    place_moe_regions(res);
    place_sampled_inputs(res);
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype,
                                   bool   is_reserve) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.is_reserve  =*/ is_reserve,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.moe_cache   =*/ moe_cache.get(),
        /*.prec_policy =*/ &model.prec_policy,
        /*.samplers    =*/ source_core_enabled() && !is_reserve && n_outputs == 0 ? std::map<llama_seq_id, llama_sampler *>{} : sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
        /*.staged_inputs =*/ use_sampled_input_async && !is_reserve && ubatch.n_tokens == 1 ? staged_inputs.get() : nullptr,
        /*.draft_vocab =*/ mtp_draft_vocab.get(),
    };
}

bool llama_context::make_graph_execution_certificate(
        const llama_ubatch * ubatch, const llama_graph_execution_intent * execution_intent,
        bool required_grouped_supported, ggml_graph_execution_certificate & certificate) const {
    certificate = {};
    const bool auxiliary = cparams.ctx_type == LLAMA_CONTEXT_TYPE_DRAFT || cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP;
    const uint32_t flags = required_grouped_execution_flags(model.moe_expert_cache_slots(), required_grouped_supported);
    if (moe_hybrid_required && source_core_enabled() && ubatch && ubatch->n_tokens > 1 && !execution_intent && !auxiliary &&
            flags == GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE) {
        LLAMA_LOG_ERROR("%s: required hybrid prefill has no supported grouped owner\n", __func__);
        return false;
    }
    uint32_t domain = GGML_GRAPH_EXECUTION_DOMAIN_INVALID;
    uint32_t row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INVALID;
    uint32_t certificate_flags = GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE;
    if (execution_intent) {
        if (!ubatch || !ubatch_matches_graph_execution_intent(
                cparams.ctx_type, model.moe_expert_cache_slots(), *ubatch, *execution_intent)) {
            LLAMA_LOG_ERROR("%s: ubatch does not match the validated execution intent\n", __func__);
            return false;
        }
        domain = execution_intent->domain;
        row_semantics = execution_intent->row_semantics;
        certificate_flags = flags;
    } else if (ubatch && ubatch_has_independent_rows(*ubatch)) {
        domain = auxiliary ? speculative_execution_domain(cparams.ctx_type) : GGML_GRAPH_EXECUTION_DOMAIN_MAIN;
        row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_INDEPENDENT;
        certificate_flags = auxiliary ? flags : GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE;
    } else if (ubatch && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT && ubatch_has_sequential_spans(*ubatch)) {
        domain = GGML_GRAPH_EXECUTION_DOMAIN_MAIN;
        row_semantics = GGML_GRAPH_EXECUTION_ROW_SEMANTICS_SEQUENTIAL;
        if (moe_hybrid_required && source_core_enabled()) { certificate_flags = flags; }
    } else {
        if (auxiliary && flags != GGML_GRAPH_EXECUTION_CERTIFICATE_FLAG_NONE) {
            LLAMA_LOG_ERROR("%s: unsupported speculative grouped MoE execution shape\n", __func__);
            return false;
        }
        return true;
    }
    certificate = {GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC, GGML_GRAPH_EXECUTION_CERTIFICATE_VERSION,
        sizeof(certificate), certificate_flags, domain, row_semantics, ubatch->n_tokens, ubatch->n_seqs_unq,
        graph_execution_owner_namespace, graph_execution_owner_generation, 0, 0, {}};
    return true;
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched,
    const llama_ubatch * ubatch,
    const llama_graph_execution_intent * execution_intent) {
    if (execution_intent != nullptr && !moe_hybrid_execution_supported(
            moe_hybrid_required, execution_intent->domain, execution_intent->row_semantics)) {
        LLAMA_LOG_ERROR("%s: required MAIN speculative hybrid execution is not supported\n", __func__);
        return GGML_STATUS_FAILED;
    }
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    copy_experts.reset();

    if (shared_workspace_peer() != nullptr) {
        workspace_in_flight = true;
    }

    ggml_graph_execution_certificate certificate = {};
    if (!make_graph_execution_certificate(ubatch, execution_intent, moe_graph_supports_required_grouped(gf), certificate)) {
        return GGML_STATUS_FAILED;
    }
    std::vector<uint8_t> phases;
    const auto & selection = ggml_moe_fidelity_selection();
    if (certificate.magic == GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC && ubatch && (selection.keep_ranks || selection.tune_misses)) {
        static_assert(int(LLAMA_BATCH_PHASE_UNKNOWN) == int(GGML_GRAPH_EXECUTION_PHASE_UNKNOWN) &&
            int(LLAMA_BATCH_PHASE_PROMPT) == int(GGML_GRAPH_EXECUTION_PHASE_PROMPT) &&
            int(LLAMA_BATCH_PHASE_GENERATION) == int(GGML_GRAPH_EXECUTION_PHASE_GENERATION), "phase ABI");
        phases.assign(ubatch->n_tokens, GGML_GRAPH_EXECUTION_PHASE_UNKNOWN);
        if (ubatch->phase) { std::copy_n(ubatch->phase, ubatch->n_tokens, phases.data()); }
        for (const auto phase : phases) { if (phase > GGML_GRAPH_EXECUTION_PHASE_GENERATION) { return GGML_STATUS_FAILED; } }
        if (use_sampled_input || execution_intent) {
            if (std::find(phases.begin(), phases.end(), GGML_GRAPH_EXECUTION_PHASE_PROMPT) != phases.end()) { return GGML_STATUS_FAILED; }
            std::fill(phases.begin(), phases.end(), GGML_GRAPH_EXECUTION_PHASE_GENERATION);
        }
    }
    const auto compute = [&]() {
        auto * res = gf_res_prev_active;
        const bool owned_result = res && res->get_gf() == gf && source_core_enabled();
        if (owned_result && res->get_source_program()) {
            return ggml_backend_sched_moe_source_program_compute_with_phases_v1(sched.get(), res->get_source_program(),
                &certificate, phases.empty() ? nullptr : phases.data(), phases.size());
        }
        const auto status = certificate.magic == GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC ?
            ggml_backend_sched_graph_compute_async_with_phases(sched.get(), gf, &certificate, phases.empty() ? nullptr : phases.data(), phases.size()) :
            ggml_backend_sched_graph_compute_async(sched.get(), gf);
        if (status == GGML_STATUS_SUCCESS && owned_result && certificate.magic == GGML_GRAPH_EXECUTION_CERTIFICATE_MAGIC) {
            uint64_t program = 0;
            const auto bound = ggml_backend_sched_moe_source_program_bind_v1(sched.get(), gf, &certificate, &program);
            if (bound == GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_OK) { res->set_source_program(program); }
            else if (bound != GGML_BACKEND_MOE_CPU_REGION_STATUS_V1_UNSUPPORTED_OPERATION) {
                (void) ggml_backend_sched_moe_source_close_v1(sched.get());
                return GGML_STATUS_FAILED;
            }
        }
        return status;
    };
    ggml_status status;
    if (moe_test_hook_set && moe_test_hook) {
        if (!ubatch || moe_test_frame || moe_test_submission == UINT64_MAX) { return GGML_STATUS_FAILED; }
        const llama_moe_test_frame frame{ubatch, &certificate, ++moe_test_submission, batched};
        struct frame_scope {
            const llama_moe_test_frame * & slot;
            ~frame_scope() { slot = nullptr; }
        } scope{moe_test_frame};
        moe_test_frame = &frame;
        status = compute();
    } else {
        status = compute();
    }
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

bool llama_context::sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);

    // the slot maps of the MoE cache
    if (lctx->moe_cache && lctx->moe_cache->copy(backend, src, dst, graph)) {
        return true;
    }

    auto & st = lctx->copy_experts;

    // the ids must be computed before the split starts, so only the first node of the split is considered
    if (ggml_graph_n_nodes(graph) == 0) {
        return false;
    }
    const ggml_tensor * node = ggml_graph_node(graph, 0);
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != dst) {
        return false;
    }

    const ggml_tensor * ids = node->src[2];
    if (ggml_nelements(ids) == 0) {
        return true;
    }

    const int64_t n_expert    = src->ne[2];
    const size_t  expert_size = src->nb[2];

    if (ids != st.ids || (int64_t) st.used.size() != n_expert) {
        st.ids_data.resize(ggml_nbytes(ids)/sizeof(int32_t));
        ggml_backend_tensor_get_async(backend, ids, st.ids_data.data(), 0, ggml_nbytes(ids));
        ggml_backend_synchronize(backend);

        st.used.assign(n_expert, false);
        for (int64_t i1 = 0; i1 < ids->ne[1]; i1++) {
            for (int64_t i0 = 0; i0 < ids->ne[0]; i0++) {
                const int32_t id = st.ids_data[i1*ids->nb[1]/sizeof(int32_t) + i0*ids->nb[0]/sizeof(int32_t)];
                GGML_ASSERT(id >= 0 && id < n_expert);
                st.used[id] = true;
            }
        }

        st.ids = ids;
    }

    // group consecutive experts and copy them together
    for (int64_t first = 0; first < n_expert; ) {
        if (!st.used[first]) {
            first++;
            continue;
        }
        int64_t last = first;
        while (last + 1 < n_expert && st.used[last + 1]) {
            last++;
        }

        // the experts in the MoE cache are copied from device memory, the others are uploaded
        int64_t next = first;
        for (int64_t e = first; e <= last && lctx->moe_cache; ) {
            const int64_t n = lctx->moe_cache->copy_experts(backend, src, dst, e, last);
            if (n == 0) {
                e++;
                continue;
            }
            if (next < e) {
                ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + next*expert_size, next*expert_size, (e - next)*expert_size);
            }
            e   += n;
            next = e;
        }

        // copy a bit extra to ensure there are no NaNs in the padding of the last expert, this is necessary for MMQ in the CUDA backend
        const size_t offset  = next*expert_size;
        const size_t padding = last < n_expert - 1 ? std::min<size_t>(expert_size, 512) : 0;
        const size_t size    = (last + 1 - next)*expert_size + padding;
        if (size > 0) {
            ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + offset, offset, size);
        }

        first = last + 1;
    }

    return true;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void discard() override {
        rinfos.clear();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            const int64_t n = rinfo.size/ggml_element_size(rinfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            if (mbuf_cur.n_tensors == mbuf.n_tensors) {
                // an equal tensor count does not imply the same chunking, e.g. save ranges [2,1] vs restore runs [1,2]
                bool same_chunking = true;
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    if (ggml_nbytes(mbuf_cur.cpy[i]) != ggml_nbytes(mbuf.org[i])) {
                        same_chunking = false;
                        break;
                    }
                }

                if (same_chunking) {
                    // same chunking: copy 1:1 by index
                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
                    }
                    continue;
                }
            }

            // different chunking: copy the write-side data (mbuf_cur.cpy) into the read-side targets (mbuf.org)
            // with a byte cursor. Write and read enumerate the same logical data in the same order but may chunk
            // it differently (even with an equal number of tensors), so copy across tensor boundaries rather than
            // 1:1 by index.
            const size_t total = mbuf_cur.total_size;

            ggml_init_params params_scratch = {
                /*.mem_size   =*/ 2*(mbuf_cur.cpy.size() + mbuf.org.size())*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx_scratch = ggml_init(params_scratch);

            size_t src_pos  = 0;
            size_t dst_pos  = 0;
            size_t src_j    = 0;
            size_t dst_i    = 0;
            size_t src_base = 0;
            size_t dst_base = 0;

            while (src_pos < total) {
                const auto & src_t = mbuf_cur.cpy[src_j];
                const auto & dst_t = mbuf.org[dst_i];

                const size_t src_size = ggml_nbytes(src_t);
                const size_t dst_size = ggml_nbytes(dst_t);

                const size_t src_off  = src_pos - src_base;
                const size_t dst_off  = dst_pos - dst_base;

                const size_t n_copy = std::min(src_size - src_off, dst_size - dst_off);

                const size_t   el   = ggml_element_size(src_t);
                const int64_t n_el = (int64_t) (n_copy / el);

                auto * src_v = ggml_view_1d(ctx_scratch, src_t, n_el, src_off);
                ggml_backend_view_init(src_v);
                auto * dst_v = ggml_view_1d(ctx_scratch, dst_t, n_el, dst_off);
                ggml_backend_view_init(dst_v);

                ggml_backend_tensor_copy(src_v, dst_v);

                src_pos += n_copy;
                dst_pos += n_copy;

                if (src_pos - src_base == src_size) {
                    src_base = src_pos;
                    ++src_j;
                }
                if (dst_pos - dst_base == dst_size) {
                    dst_base = dst_pos;
                    ++dst_i;
                }
            }

            GGML_ASSERT(src_pos == total && dst_pos == total);
            // any tensors left unvisited hold no data
            for (size_t i = src_j; i < mbuf_cur.cpy.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf_cur.cpy[i]) == 0);
            }
            for (size_t i = dst_i; i < mbuf.org.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf.org[i]) == 0);
            }

            ggml_free(ctx_scratch);
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    void discard() override {
        rinfos.clear();
        buf_size = 0;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io.discard();
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io->discard();
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (tokens_out == nullptr) {
            const size_t n_token_max = (file.size() - file.tell()) / sizeof(llama_token);
            if (n_token_count > n_token_max) {
                LLAMA_LOG_ERROR("%s: token count in sequence state file exceeds the file size! %u > %zu\n", __func__, n_token_count, n_token_max);
                return 0;
            }

            *n_token_count_out = n_token_count;
            return file.tell();
        }

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    for (const auto & [buft, size] : model.moe_expert_cache_memory_breakdown(cparams.ctx_type)) {
        ret[buft].context += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (moe_cache) {
        for (const auto & [buft, size] : moe_cache->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    if (cparams.flash_attn) {
        LLAMA_LOG_INFO("%s: disabling flash attention, FLASH_ATTN_EXT has no backward pass\n", __func__);
        cparams.flash_attn = false;
    }

    // gradients cannot flow through the KV cache, so the attention reads the K and V of the current ubatch directly
    if (n_ubatch == cparams.n_ctx) {
        cparams.training = true;
    } else {
        LLAMA_LOG_WARN("%s: n_ubatch (%u) != n_ctx (%u), the K and V projections will not receive gradients\n", __func__, n_ubatch, cparams.n_ctx);
    }

    // the training graph is different, need to reserve again
    request_sched_reserve();
    sched_reserve();

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    if (reset_source_core_checked() != GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK) { return; }
    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        // TODO: use llama_batch_ext here
        {
            llama_batch_compat compat(this, batch);
            if (!balloc->init(*compat.batch_ext, model.vocab, true)) {
                LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
                return;
            }
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = get_gf_res_prev();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            // the optimizer graph is allocated outside sched, so the next decode must rebuild
            gf_res_prev_active = nullptr;
            if (reset_source_core_checked() != GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK) { break; }
            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            copy_experts.reset();
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_outputs_max_per_seq       =*/ 1,
        /*.kv_gpu_layers               =*/ 0,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.moe_cache_size              =*/ 0,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.kv_cpu_pinned               =*/ false,
        /*.recurrent_state_offload     =*/ false,
        /*.phase_aware_workspace       =*/ false,
        /*.moe_source_graph_capacity   =*/ false,
        /*.live_context_workspace      =*/ false,
        /*.decode_boundary_overlap     =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
    };

    return result;
}

static llama_context * llama_init_from_model_impl(
                 llama_model * model,
        llama_context_params   params,
                 const char * profile_path,
                 const char * profile_adaptation, const char * cache_allocation = "auto") {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (model->get_split_state_ud.n_devices == 1) {
            LLAMA_LOG_WARN("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", __func__);
        }
    }

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params, profile_path, profile_adaptation, cache_allocation);
        const auto & cparams = ctx->get_cparams();

        if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN && cparams.rope_freq_scale != model->hparams.rope_freq_scale_train) {
            LLAMA_LOG_INFO("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", __func__, model->hparams.n_ctx_train);
            model->hparams.n_ctx_train = cparams.n_ctx_orig_yarn / cparams.rope_freq_scale;
            LLAMA_LOG_INFO("%s: n_ctx_train adjusted to %u\n", __func__, model->hparams.n_ctx_train);
        }

        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

llama_context * llama_init_from_model(llama_model * model, llama_context_params params) {
    return llama_init_from_model_impl(model, params, nullptr, nullptr);
}

llama_context * llama_init_from_model_with_moe_profile(llama_model * model, llama_context_params params,
        const char * path, const char * adaptation) {
    if (!path || !*path) {
        LLAMA_LOG_ERROR("%s: profile path cannot be empty\n", __func__);
        return nullptr;
    }
    return llama_init_from_model_impl(model, params, path, adaptation);
}

llama_context * llama_init_from_model_with_moe_cache_policy(llama_model * model, llama_context_params params,
        const char * path, const char * adaptation, const char * allocation) {
    return llama_init_from_model_impl(model, params, path, adaptation, allocation);
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

bool llama_set_mtp_draft_vocab(llama_context * ctx, const char * path) {
    return ctx && ctx->set_mtp_draft_vocab(path);
}

bool llama_write_mtp_draft_vocab(const llama_model * model, const int32_t * ids, size_t count, const char * path) {
    try {
        return model && llama_draft_vocab_write(model->vocab, ids, count, path);
    } catch (const std::exception & error) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, error.what());
        return false;
    }
}

int32_t llama_moe_source_context_close_v1(llama_context * ctx) {
    return ctx ? ctx->close_source_core_checked() : GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID;
}

int32_t llama_moe_source_context_free_v1(llama_context ** ctx) {
    if (!ctx) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_INVALID; }
    if (!*ctx) { return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK; }
    const auto status = (*ctx)->close_source_core_checked();
    if (status) { return status; }
    delete *ctx;
    *ctx = nullptr;
    return GGML_BACKEND_MOE_SOURCE_CORE_STATUS_V1_OK;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

bool llama_recurrent_sparse_snapshots_supported(const llama_context * ctx) {
    return ctx != nullptr && ctx->recurrent_sparse_snapshots_supported();
}

bool llama_recurrent_set_sparse_snapshot_mode(llama_context * ctx, bool enabled, int32_t selected_token) {
    if (ctx == nullptr || ctx->get_memory() == nullptr || (enabled && !ctx->recurrent_sparse_snapshots_supported())) {
        return false;
    }
    return ctx->get_memory()->recurrent_set_sparse_snapshot_mode(enabled, selected_token);
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

bool llama_get_causal_attn(const llama_context * ctx) {
    return ctx->get_causal_attn();
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

bool llama_set_ple_prefetch(llama_context * ctx, bool enabled) {
    return ctx->set_ple_prefetch(enabled);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

bool llama_get_sampled_output_view(llama_context * ctx, int32_t i, llama_sampled_output_view * view) {
    if (!ctx || !view) {
        return false;
    }
    const auto & selection = ggml_moe_fidelity_selection();
    if (!selection.valid || !selection.source_pool) {
        return false;
    }

    ctx->synchronize();
    *view = {};
    view->probs = ctx->get_sampled_probs_ith(i);
    view->logits = ctx->get_sampled_logits_ith(i);
    view->candidates = ctx->get_sampled_candidates_ith(i);
    if (view->probs) {
        view->probs_count = static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
    } else if (view->logits) {
        view->logits_count = static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
    }
    return true;
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

// compat: llama_batch -> llama_batch_ext -> encode/decode

int llama_context::encode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp, model.hparams.n_embd_inp_enc());
    return encode(*compat.batch_ext);
}

int llama_context::decode(const llama_batch & batch_inp, const llama_decode_execution_intent * intent) {
    llama_graph_execution_intent execution;
    if (intent && !target_verification_intent_valid(intent, batch_inp, execution)) { return -1; }
    llama_batch_compat compat(this, batch_inp);
    return decode(*compat.batch_ext, intent);
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch, nullptr);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode_ext(
        llama_context * ctx,
          llama_batch   batch,
    const llama_decode_execution_intent * intent) {
    const int ret = ctx->decode(batch, intent);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode_sampled(llama_context * ctx, llama_seq_id seq_id, llama_pos pos) {
    return ctx ? ctx->decode_sampled(seq_id, pos) : -1;
}

int32_t llama_decode_sampled_async(llama_context * ctx, llama_seq_id seq_id, llama_pos pos, llama_token * previous) {
    return ctx && previous ? ctx->decode_sampled(seq_id, pos, previous) : -1;
}

int32_t llama_decode_sampled_batch_async(llama_context * ctx, const llama_sampled_decode_item * items, int32_t n_items, llama_token * previous) {
    return ctx && previous ? ctx->decode_sampled(items, n_items, previous) : -1;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    return llama_process_ext(ctx, type, batch, nullptr);
}

int32_t llama_process_ext(llama_context * ctx, llama_process_type type, llama_batch_ext * batch,
        const llama_decode_execution_intent * intent) {
    if (!ctx || !batch || (intent && type != LLAMA_PROCESS_TYPE_DECODE)) { return -1; }
    switch (type) {
        case LLAMA_PROCESS_TYPE_ENCODE: return ctx->encode(*batch);
        case LLAMA_PROCESS_TYPE_DECODE: return ctx->decode(*batch, intent);
    }
    return -1;
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}

int32_t llama_attach_shared_workspace(llama_context * borrower, llama_context * owner) {
    if (borrower == nullptr || owner == nullptr) {
        return 0;
    }
    return borrower->attach_shared_workspace(*owner);
}

bool llama_contexts_share_workspace(const llama_context * ctx_a, const llama_context * ctx_b) {
    return ctx_a != nullptr && ctx_b != nullptr && ctx_a->shares_workspace_with(*ctx_b);
}

uint64_t llama_trim_transient_memory(llama_context * ctx) {
    return ctx != nullptr ? ctx->trim_transient_memory() : 0;
}
