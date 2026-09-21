#include "fit.h"

#include "../src/llama-ext.h"
#include "common.h"
#include "hash/hash.h"
#include "json.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static const char * common_moe_mode_name(llama_moe_placement_mode mode) {
    switch (mode) {
        case LLAMA_MOE_PLACEMENT_ORDINARY_CPU:    return "ordinary_cpu";
        case LLAMA_MOE_PLACEMENT_ORDINARY_DEVICE: return "ordinary_device";
        case LLAMA_MOE_PLACEMENT_RESIDUAL_CACHE:  return "residual_cache";
        case LLAMA_MOE_PLACEMENT_MIXED:           return "mixed";
    }
    return "unknown";
}

static common_json common_moe_configuration_record(const llama_moe_placement_report & report,
                                                   const llama_model_params &         mparams,
                                                   const llama_context_params &       cparams,
                                                   const common_params *              runtime_params,
                                                   const std::string &                sampler_configuration_id,
                                                   const char *                       role) {
    GGML_UNUSED(mparams);
    common_json model = common_json::object();
    common_json model_allocations = common_json::array();
    std::vector<const llama_moe_model_allocation *> ordered_allocations;
    ordered_allocations.reserve(report.model_allocations.size());
    for (const auto & allocation : report.model_allocations) {
        ordered_allocations.push_back(&allocation);
    }
    std::sort(ordered_allocations.begin(), ordered_allocations.end(), [](const auto * lhs, const auto * rhs) {
        if (lhs->owner_canonical_id != rhs->owner_canonical_id) {
            return lhs->owner_canonical_id < rhs->owner_canonical_id;
        }
        return lhs->resolved_class < rhs->resolved_class;
    });
    for (const auto * allocation : ordered_allocations) {
        common_json item = common_json::object();
        item["bytes"] = allocation->bytes_available ? common_json_value(allocation->bytes) : common_json_value(nullptr);
        item["owner_canonical_id"] = allocation->owner_canonical_id;
        item["owner_identity_kind"] = allocation->owner_identity_kind;
        item["resolved_class"] = allocation->resolved_class;
        model_allocations.push_back(std::move(item));
    }
    model["allocations"] = std::move(model_allocations);
    model["identity"] = report.model_identity;
    model["identity_kind"] = report.model_identity_kind;
    model["placement_id"] = report.placement_id;

    common_json override_provenance = common_json::array();
    for (const auto & group : report.groups) {
        for (const auto & bank : group.banks) {
            common_json item = common_json::object();
            item["bank"] = bank.name;
            item["group"] = group.semantic_index;
            item["origin"] = static_cast<int32_t>(bank.reason);
            item["requested_buft"] = bank.requested_buft;
            item["resolved_buft"] = bank.resolved_buft;
            item["selected_buft"] = bank.selected_buft;
            item["winning_override_index"] = bank.winning_override_index;
            override_provenance.push_back(std::move(item));
        }
    }

    common_json context = common_json::object();
    context["n_ctx"] = cparams.n_ctx;
    context["n_batch"] = cparams.n_batch;
    context["n_ubatch"] = cparams.n_ubatch;
    context["n_seq_max"] = cparams.n_seq_max;
    context["n_rs_seq"] = cparams.n_rs_seq;
    context["n_outputs_max"] = cparams.n_outputs_max;
    context["n_outputs_max_per_seq"] = cparams.n_outputs_max_per_seq;
    context["kv_gpu_layers"] = cparams.kv_gpu_layers;
    context["n_threads"] = cparams.n_threads;
    context["n_threads_batch"] = cparams.n_threads_batch;
    context["ctx_type"] = static_cast<int32_t>(cparams.ctx_type);
    context["rope_scaling_type"] = static_cast<int32_t>(cparams.rope_scaling_type);
    context["pooling_type"] = static_cast<int32_t>(cparams.pooling_type);
    context["attention_type"] = static_cast<int32_t>(cparams.attention_type);
    context["flash_attn_type"] = static_cast<int32_t>(cparams.flash_attn_type);
    context["rope_freq_base"] = cparams.rope_freq_base;
    context["rope_freq_scale"] = cparams.rope_freq_scale;
    context["yarn_ext_factor"] = cparams.yarn_ext_factor;
    context["yarn_attn_factor"] = cparams.yarn_attn_factor;
    context["yarn_beta_fast"] = cparams.yarn_beta_fast;
    context["yarn_beta_slow"] = cparams.yarn_beta_slow;
    context["yarn_orig_ctx"] = cparams.yarn_orig_ctx;
    context["type_k"] = static_cast<int32_t>(cparams.type_k);
    context["type_v"] = static_cast<int32_t>(cparams.type_v);
    context["embeddings"] = cparams.embeddings;
    context["offload_kqv"] = cparams.offload_kqv;
    context["no_perf"] = cparams.no_perf;
    context["op_offload"] = cparams.op_offload;
    context["swa_full"] = cparams.swa_full;
    context["kv_unified"] = cparams.kv_unified;
    context["kv_cpu_pinned"] = cparams.kv_cpu_pinned;
    context["recurrent_state_offload"] = cparams.recurrent_state_offload;
    context["phase_aware_workspace"] = cparams.phase_aware_workspace;
    context["live_context_workspace"] = cparams.live_context_workspace;
    context["decode_boundary_overlap"] = cparams.decode_boundary_overlap;
    common_json sampler                = common_json::object();
    sampler["configuration_id"] =
        cparams.n_samplers == 0 ? common_json_value(nullptr) : common_json_value(sampler_configuration_id);
    common_json                             sampler_sequences = common_json::array();
    std::map<const llama_sampler *, size_t> sampler_aliases;
    for (size_t i = 0; i < cparams.n_samplers; ++i) {
        const auto & config = cparams.samplers[i];
        size_t       alias  = 0;
        if (config.sampler != nullptr) {
            auto [it, inserted] = sampler_aliases.emplace(config.sampler, sampler_aliases.size());
            GGML_UNUSED(inserted);
            alias = it->second + 1;
        }
        sampler_sequences.push_back({
            { "sampler_alias", alias         },
            { "seq_id",        config.seq_id }
        });
    }
    sampler["sequences"] = std::move(sampler_sequences);
    context["samplers"]  = std::move(sampler);

    const auto env_present = [](const char * name) { return std::getenv(name) != nullptr; };
    const auto env_int = [](const char * name, int fallback) {
        const char * value = std::getenv(name);
        return value != nullptr ? std::atoi(value) : fallback;
    };
    const auto env_string = [](const char * name, const char * fallback) {
        const char * value = std::getenv(name);
        return std::string(value != nullptr ? value : fallback);
    };
    bool early_router_env = env_string("GGML_CUDA_MOE_EARLY_ROUTER", "0") == "1" &&
        env_string("GGML_CUDA_MOE_EARLY_ROUTER_LOOKAHEAD", "1") == "1";
    for (const char * suffix : {
             "NATIVE", "COPY_ENGINE", "COPY_MAILBOX", "COPY_POLL", "COPY_BATCH", "COPY_SPLIT",
             "COPY_READY_ONLY", "COPY_BANKS", "COPY_DEBUG", "STAGE_BLOCKS"}) {
        const std::string name = std::string("GGML_CUDA_MOE_EARLY_ROUTER_") + suffix;
        const char * value = std::getenv(name.c_str());
        early_router_env = early_router_env && (value == nullptr || std::string(value) == "0");
    }
    common_json environment = common_json::object();
    const std::string allreduce = env_string("GGML_CUDA_ALLREDUCE",
#if defined(__linux__)
        "nccl"
#else
        "internal"
#endif
    );
    environment["allreduce"] = allreduce == "nccl" || allreduce == "internal" || allreduce == "none" ?
        allreduce : "none";
    std::string cublas_compute_type = env_string("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "auto");
    std::transform(cublas_compute_type.begin(), cublas_compute_type.end(), cublas_compute_type.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (cublas_compute_type == "fp32") {
        cublas_compute_type = "f32";
    } else if (cublas_compute_type == "fp16") {
        cublas_compute_type = "f16";
    } else if (cublas_compute_type != "auto" && cublas_compute_type != "f32" &&
               cublas_compute_type != "f16" && cublas_compute_type != "bf16") {
        cublas_compute_type = "auto";
    }
    environment["cublas_compute_type"] = cublas_compute_type;
    environment["disable_fusion"] = env_int("GGML_CUDA_DISABLE_FUSION", 0) != 0;
    environment["disable_graphs"] = env_present("GGML_CUDA_DISABLE_GRAPHS");
    environment["enable_unified_memory"] = env_present("GGML_CUDA_ENABLE_UNIFIED_MEMORY");
    environment["graph_optimization"] = env_int("GGML_CUDA_GRAPH_OPT", 0) == 1;
    environment["graph_profile"] = env_present("GGML_CUDA_GRAPH_PROFILE");
    environment["moe_early_router_env"] = early_router_env;
    environment["moe_frequency"] = env_int("GGML_CUDA_MOE_FREQUENCY", 1) != 0;
    environment["no_pinned"] = env_present("GGML_CUDA_NO_PINNED");
    environment["op_offload_min_batch"] = env_int("GGML_OP_OFFLOAD_MIN_BATCH", 32);
    environment["p2p"] = env_present("GGML_CUDA_P2P");
    environment["pdl"] = env_int("GGML_CUDA_PDL", 1) != 0;
    environment["register_host"] = env_present("GGML_CUDA_REGISTER_HOST");

    common_json runtime = common_json::object();
    if (runtime_params != nullptr) {
        runtime["backend_sampling"] = runtime_params->sampling.backend_sampling;
        runtime["decode_overlap"] = runtime_params->decode_overlap;
        runtime["moe_early_router"] = runtime_params->moe_early_router || early_router_env;
        runtime["ple_prefetch"] = runtime_params->ple_prefetch;
        runtime["sampling_seed"] = runtime_params->sampling.seed;
        runtime["sampling_temperature"] = runtime_params->sampling.temp;
        runtime["sampling_top_k"] = runtime_params->sampling.top_k;
        runtime["sampling_top_p"] = runtime_params->sampling.top_p;
        runtime["spec_draft_backend_sampling"] = runtime_params->speculative.draft.backend_sampling;
        runtime["spec_draft_kv_gpu_layers"] = runtime_params->speculative.draft.kv_gpu_layers;
        runtime["spec_draft_n_max"] = runtime_params->speculative.draft.n_max;
        runtime["spec_draft_n_min"] = runtime_params->speculative.draft.n_min;
        runtime["spec_draft_n_ubatch"] = runtime_params->speculative.draft.n_ubatch;
        runtime["spec_draft_p_min"] = runtime_params->speculative.draft.p_min;
        runtime["spec_draft_p_split"] = runtime_params->speculative.draft.p_split;
        common_json spec_types = common_json::array();
        for (const auto type : runtime_params->speculative.types) {
            spec_types.push_back(static_cast<int32_t>(type));
        }
        runtime["speculative_types"] = std::move(spec_types);
    } else {
        runtime["moe_early_router"] = early_router_env;
    }

    common_json configuration = common_json::object();
    configuration["context"] = std::move(context);
    configuration["environment"] = std::move(environment);
    configuration["model"] = std::move(model);
    configuration["override_provenance"] = std::move(override_provenance);
    configuration["placement"] = common_json::parse(report.placement_record);
    configuration["role"]                = role;
    configuration["runtime"] = std::move(runtime);
    return configuration;
}

std::string common_moe_placement_report_json(
        const llama_moe_placement_report & report,
        const llama_model_params & mparams,
        const llama_context_params & cparams,
        const common_params * runtime_params) {
    common_json configuration = common_moe_configuration_record(report, mparams, cparams, runtime_params, "", "target");
    const std::string configuration_record = configuration.dump();
    const std::string configuration_id = hash_sha256_hex(configuration_record.data(), configuration_record.size());

    common_json owners = common_json::array();
    for (const auto & owner : report.owners) {
        common_json item = common_json::object();
        item["selected_index"] = owner.selected_index;
        item["canonical_id"] = owner.canonical_id;
        item["identity_kind"] = owner.identity_kind;
        item["backend"] = owner.backend;
        item["name"] = owner.name;
        item["description"] = owner.description;
        item["slots"] = owner.slots;
        item["cache_byte_cap"] = owner.cache_byte_cap;
        item["active_groups"] = owner.active_groups;
        item["active_context_mask"] = owner.active_context_mask;
        item["active_cache_groups"] = owner.active_cache_groups;
        item["active_cache_context_mask"] = owner.active_cache_context_mask;
        item["cache_fixed_bytes"] = owner.cache_fixed_bytes;
        item["cache_per_slot_bytes"] = owner.cache_per_slot_bytes;
        item["mandatory_host_staging_default_bytes"] = owner.mandatory_host_staging_available ?
            common_json_value(owner.mandatory_host_staging_default_bytes) : common_json_value(nullptr);
        item["mandatory_host_staging_mtp_bytes"] = owner.mandatory_host_staging_available ?
            common_json_value(owner.mandatory_host_staging_mtp_bytes) : common_json_value(nullptr);
        item["mandatory_host_staging_provenance"] = owner.mandatory_host_staging_available ?
            common_json_value(owner.mandatory_host_staging_provenance) : common_json_value(nullptr);
        owners.push_back(std::move(item));
    }

    common_json groups = common_json::array();
    for (const auto & group : report.groups) {
        common_json item = common_json::object();
        item["semantic_index"] = group.semantic_index;
        item["layer"] = group.layer;
        item["layout"] = group.layout;
        item["domain"] = group.domain;
        item["n_experts"] = group.n_experts;
        item["top_k"] = group.top_k;
        item["context_use_mask"] = group.context_use_mask;
        item["mode"] = common_moe_mode_name(group.mode);
        item["owner_canonical_id"] = group.owner_canonical_id.empty() ?
            common_json_value(nullptr) : common_json_value(group.owner_canonical_id);
        item["cache_owner_canonical_id"] = group.cache_owner_canonical_id.empty() ?
            common_json_value(nullptr) : common_json_value(group.cache_owner_canonical_id);
        item["placement_reason"] = group.placement_reason;
        item["ordinary_allocation_estimate"] = group.ordinary_allocation_estimate;
        item["ordinary_allocation_provenance"] = group.ordinary_allocation_provenance;
        item["cache_fixed_bytes"] = group.cache_fixed_bytes;
        item["cache_per_slot_bytes"] = group.cache_per_slot_bytes;
        item["cache_sizing_provenance"] = group.cache_sizing_provenance.empty() ?
            common_json_value(nullptr) : common_json_value(group.cache_sizing_provenance);
        common_json banks = common_json::array();
        for (const auto & bank : group.banks) {
            common_json bank_json = common_json::object();
            bank_json["name"] = bank.name;
            bank_json["role"] = bank.role;
            bank_json["status"] = bank.status;
            bank_json["type"] = bank.type;
            bank_json["expert_stride"] = bank.expert_stride;
            bank_json["tensor_bytes"] = bank.tensor_bytes;
            bank_json["mode"] = common_moe_mode_name(bank.mode);
            bank_json["owner_canonical_id"] = bank.owner_canonical_id.empty() ?
                common_json_value(nullptr) : common_json_value(bank.owner_canonical_id);
            bank_json["owner_identity_kind"] = bank.owner_identity_kind;
            bank_json["resolved_class"] = bank.resolved_class;
            bank_json["actual_buft"] = bank.actual_buft_available ?
                common_json_value(bank.actual_buft) : common_json_value(nullptr);
            bank_json["winning_override_index"] = bank.winning_override_index;
            bank_json["winning_override_pattern_redacted"] = bank.winning_override_index >= 0;
            bank_json["requested_buft"] = bank.requested_buft;
            bank_json["selected_buft"] = bank.selected_buft;
            bank_json["resolved_buft"] = bank.resolved_buft;
            banks.push_back(std::move(bank_json));
        }
        item["banks"] = std::move(banks);
        groups.push_back(std::move(item));
    }

    common_json allocations = common_json::array();
    for (const auto & allocation : report.model_allocations) {
        common_json item = common_json::object();
        item["resolved_class"] = allocation.resolved_class;
        item["owner_canonical_id"] = allocation.owner_canonical_id;
        item["owner_identity_kind"] = allocation.owner_identity_kind;
        item["owner_backend"] = allocation.owner_backend;
        item["bytes"] = allocation.bytes_available ? common_json_value(allocation.bytes) : common_json_value(nullptr);
        item["current_allocation"] = allocation.current_allocation;
        item["provenance"] = allocation.provenance;
        allocations.push_back(std::move(item));
    }

    common_json shared = common_json::array();
    for (const auto & tensor : report.shared_tensors) {
        common_json item = common_json::object();
        item["name"] = tensor.name;
        item["tensor_bytes"] = tensor.tensor_bytes;
        item["type"] = tensor.type;
        item["resolved_class"] = tensor.resolved_storage_available ?
            common_json_value(tensor.resolved_class) : common_json_value(nullptr);
        item["owner_canonical_id"] = tensor.resolved_storage_available ?
            common_json_value(tensor.owner_canonical_id) : common_json_value(nullptr);
        item["owner_identity_kind"] = tensor.resolved_storage_available ?
            common_json_value(tensor.owner_identity_kind) : common_json_value(nullptr);
        item["storage_relation"] = tensor.storage_relation;
        item["current_storage"] = tensor.current_storage_available;
        item["storage_provenance"] = tensor.storage_provenance;
        shared.push_back(std::move(item));
    }

    common_json root = common_json::object();
    root["schema_version"] = report.schema_version;
    root["report_kind"] = "moe_placement";
    root["configuration_id"] = configuration_id;
    root["configuration"] = std::move(configuration);
    root["placement_id"] = report.placement_id;
    root["placement_identity_kind"] = report.placement_identity_kind;
    root["model_identity"] = report.model_identity;
    root["model_identity_kind"] = report.model_identity_kind;
    root["measurement_completeness"] = "incomplete";
    root["capacity_assessment"] = "unknown";
    root["diagnostics"] = common_json::array({"runtime observations and capacity limits were not measured"});
    root["owners"] = std::move(owners);
    root["groups"] = std::move(groups);
    root["model_allocations"] = std::move(allocations);
    root["shared_tensors"] = std::move(shared);
    return root.dump();
}

std::string common_moe_placement_report_human(
        const llama_moe_placement_report & report,
        const llama_model_params & mparams,
        const llama_context_params & cparams,
        const common_params * runtime_params) {
    const common_json machine = common_json::parse(
        common_moe_placement_report_json(report, mparams, cparams, runtime_params));
    std::ostringstream out;
    out << "MoE placement " << machine.at("configuration_id").get<std::string>() << '\n'
        << "  placement: " << report.placement_id << " (" << report.placement_identity_kind << ")\n"
        << "  model: " << report.model_identity << " (" << report.model_identity_kind << ")\n"
        << "  role: target\n"
        << "  status: incomplete; capacity unknown (runtime observations not measured)\n"
        << "  owners: " << report.owners.size() << ", groups: " << report.groups.size() << '\n';
    for (const auto & owner : report.owners) {
        out << "    [" << owner.selected_index << "] " << owner.name
            << " id=" << owner.canonical_id << " slots=" << owner.slots
            << " cache=" << owner.cache_fixed_bytes << "+" << owner.cache_per_slot_bytes << "/slot"
            << " active_groups=" << owner.active_groups << " cached_groups=" << owner.active_cache_groups
            << " staging=";
        if (owner.mandatory_host_staging_available) {
            out << owner.mandatory_host_staging_default_bytes << "+" << owner.mandatory_host_staging_mtp_bytes;
        } else {
            out << "unknown";
        }
        out << '\n';
    }
    struct group_summary {
        size_t count = 0;
        int32_t first_layer = std::numeric_limits<int32_t>::max();
        int32_t last_layer = std::numeric_limits<int32_t>::min();
        size_t cache_fixed = 0;
        size_t cache_per_slot = 0;
        size_t ordinary_estimate = 0;
    };
    std::map<std::string, group_summary> summaries;
    for (const auto & group : report.groups) {
        const std::string owner = group.owner_canonical_id.empty() ? "mixed/unowned" : group.owner_canonical_id;
        const std::string cache_owner = group.cache_owner_canonical_id.empty() ? "none" : group.cache_owner_canonical_id;
        const std::string key = std::string(common_moe_mode_name(group.mode)) + "|" + owner + "|" +
            cache_owner + "|" + std::to_string(group.context_use_mask);
        auto & summary = summaries[key];
        ++summary.count;
        summary.first_layer = std::min(summary.first_layer, group.layer);
        summary.last_layer = std::max(summary.last_layer, group.layer);
        summary.cache_fixed += group.cache_fixed_bytes;
        summary.cache_per_slot += group.cache_per_slot_bytes;
        summary.ordinary_estimate += group.ordinary_allocation_estimate;
    }
    for (const auto & [key, summary] : summaries) {
        out << "    group_set " << key << " layers=" << summary.first_layer << '-' << summary.last_layer
            << " count=" << summary.count << " cache=" << summary.cache_fixed << '+'
            << summary.cache_per_slot << "/slot ordinary_estimate=" << summary.ordinary_estimate << '\n';
    }
    for (const auto & allocation : report.model_allocations) {
        out << "    model_buffer " << allocation.resolved_class << " bytes=";
        if (allocation.bytes_available) {
            out << allocation.bytes;
        } else {
            out << "unknown";
        }
        out << " provenance=" << allocation.provenance << '\n';
    }
    for (const auto & tensor : report.shared_tensors) {
        out << "    shared_tensor " << tensor.name << " bytes=" << tensor.tensor_bytes
            << " relation=" << tensor.storage_relation << " owner="
            << (tensor.resolved_storage_available ? tensor.owner_canonical_id : "unknown") << '\n';
    }
    return out.str();
}

static const char * common_joint_role_name(common_joint_role role) {
    switch (role) {
        case COMMON_JOINT_ROLE_TARGET:
            return "target";
        case COMMON_JOINT_ROLE_DRAFT:
            return "draft";
        case COMMON_JOINT_ROLE_MTP:
            return "mtp";
    }
    return "unknown";
}

static const char * common_joint_sharing_name(common_joint_sharing sharing) {
    switch (sharing) {
        case COMMON_JOINT_SHARING_NONE:
            return "none";
        case COMMON_JOINT_SHARING_BORROW_TARGET:
            return "borrow_target";
        case COMMON_JOINT_SHARING_TARGET_MODEL:
            return "target_model";
    }
    return "unknown";
}

static const char * common_joint_completeness_name(common_joint_completeness completeness) {
    switch (completeness) {
        case COMMON_JOINT_COMPLETENESS_COMPLETE:
            return "complete";
        case COMMON_JOINT_COMPLETENESS_INCOMPLETE:
            return "incomplete";
        case COMMON_JOINT_COMPLETENESS_ERROR:
            return "error";
    }
    return "error";
}

static const char * common_joint_capacity_name(common_joint_capacity capacity) {
    switch (capacity) {
        case COMMON_JOINT_CAPACITY_WITHIN_LIMITS:
            return "within_limits";
        case COMMON_JOINT_CAPACITY_EXCEEDS_LIMITS:
            return "exceeds_limits";
        case COMMON_JOINT_CAPACITY_UNKNOWN:
            return "unknown";
    }
    return "unknown";
}

static const char * common_joint_bound_name(common_joint_bound bound) {
    switch (bound) {
        case COMMON_JOINT_BOUND_EXACT_ESTIMATE:
            return "exact_estimate";
        case COMMON_JOINT_BOUND_UPPER_ESTIMATE:
            return "upper_estimate";
        case COMMON_JOINT_BOUND_UNKNOWN:
            return "unknown";
    }
    return "unknown";
}

static bool checked_add_size(size_t & dst, size_t value) {
    if (value > SIZE_MAX - dst) {
        return false;
    }
    dst += value;
    return true;
}

static bool checked_add_memory(common_joint_memory & dst, const common_joint_memory & value) {
    dst.staging_available = dst.staging_available && value.staging_available;
    return checked_add_size(dst.model, value.model) && checked_add_size(dst.context, value.context) &&
           checked_add_size(dst.compute, value.compute) && checked_add_size(dst.staging, value.staging);
}

static bool memory_total(const common_joint_memory & memory, size_t & total) {
    total = 0;
    return checked_add_size(total, memory.model) && checked_add_size(total, memory.context) &&
           checked_add_size(total, memory.compute) && checked_add_size(total, memory.staging);
}

struct common_owned_model_params {
    llama_model_params                            params;
    std::vector<ggml_backend_dev_t>               devices;
    std::vector<float>                            tensor_split;
    std::vector<std::string>                      override_patterns;
    std::vector<llama_model_tensor_buft_override> overrides;
    std::vector<llama_model_kv_override>          kv_overrides;
    std::vector<llama_model_layer_range>          cache_ranges;
    std::vector<size_t>                           cache_budgets;

    explicit common_owned_model_params(const llama_model_params & source) : params(source) {
        if (source.devices != nullptr) {
            for (size_t i = 0; i < llama_max_devices() && source.devices[i] != nullptr; ++i) {
                devices.push_back(source.devices[i]);
            }
            devices.push_back(nullptr);
        }
        if (source.tensor_split != nullptr) {
            tensor_split.assign(source.tensor_split, source.tensor_split + llama_max_devices());
        }
        if (source.tensor_buft_overrides != nullptr) {
            const size_t max_overrides = llama_max_tensor_buft_overrides();
            for (size_t i = 0; i < max_overrides && source.tensor_buft_overrides[i].pattern != nullptr; ++i) {
                override_patterns.emplace_back(source.tensor_buft_overrides[i].pattern);
            }
            overrides.reserve(override_patterns.size() + 1);
            for (size_t i = 0; i < override_patterns.size(); ++i) {
                overrides.push_back({ override_patterns[i].c_str(), source.tensor_buft_overrides[i].buft });
            }
            overrides.push_back({ nullptr, nullptr });
        }
        if (source.kv_overrides != nullptr) {
            constexpr size_t max_kv_overrides = 4096;
            size_t           i                = 0;
            for (; i < max_kv_overrides && source.kv_overrides[i].key[0] != '\0'; ++i) {
                kv_overrides.push_back(source.kv_overrides[i]);
            }
            if (i == max_kv_overrides) {
                throw std::runtime_error("unterminated model metadata override array");
            }
            kv_overrides.push_back({});
        }
        if (source.moe_expert_cache_layer_ranges != nullptr && source.n_moe_expert_cache_layer_ranges > 0) {
            cache_ranges.assign(source.moe_expert_cache_layer_ranges,
                                source.moe_expert_cache_layer_ranges + source.n_moe_expert_cache_layer_ranges);
        }
        if (source.moe_expert_cache_byte_budgets != nullptr && source.n_moe_expert_cache_byte_budgets > 0) {
            cache_budgets.assign(source.moe_expert_cache_byte_budgets,
                                 source.moe_expert_cache_byte_budgets + source.n_moe_expert_cache_byte_budgets);
        }
        params.devices                         = devices.empty() ? nullptr : devices.data();
        params.tensor_split                    = tensor_split.empty() ? nullptr : tensor_split.data();
        params.tensor_buft_overrides           = overrides.empty() ? nullptr : overrides.data();
        params.kv_overrides                    = kv_overrides.empty() ? nullptr : kv_overrides.data();
        params.moe_expert_cache_layer_ranges   = cache_ranges.empty() ? nullptr : cache_ranges.data();
        params.n_moe_expert_cache_layer_ranges = cache_ranges.size();
        params.moe_expert_cache_byte_budgets   = cache_budgets.empty() ? nullptr : cache_budgets.data();
        params.n_moe_expert_cache_byte_budgets = cache_budgets.size();
        params.progress_callback               = nullptr;
        params.progress_callback_user_data     = nullptr;
        params.model_shared                    = nullptr;
        params.no_alloc                        = true;
    }
};

struct common_owned_context_params {
    llama_context_params                  params;
    std::vector<llama_sampler_ptr>        owned_samplers;
    std::vector<llama_sampler_seq_config> samplers;

    explicit common_owned_context_params(const llama_context_params & source) : params(source) {
        if (source.samplers != nullptr && source.n_samplers != 0) {
            owned_samplers.reserve(source.n_samplers);
            samplers.reserve(source.n_samplers);
            std::unordered_map<const llama_sampler *, llama_sampler *> clones;
            for (size_t i = 0; i < source.n_samplers; ++i) {
                const auto & source_config = source.samplers[i];
                if (source_config.sampler == nullptr) {
                    samplers.push_back(source_config);
                    continue;
                }
                auto [it, inserted] = clones.emplace(source_config.sampler, nullptr);
                if (inserted) {
                    llama_sampler_ptr clone(llama_sampler_clone(source_config.sampler));
                    if (clone == nullptr) {
                        throw std::runtime_error("failed to clone context sampler");
                    }
                    it->second = clone.get();
                    owned_samplers.push_back(std::move(clone));
                }
                samplers.push_back({ source_config.seq_id, it->second });
            }
        }
        params.samplers            = samplers.empty() ? nullptr : samplers.data();
        params.n_samplers          = samplers.size();
        params.cb_eval             = nullptr;
        params.cb_eval_user_data   = nullptr;
        params.abort_callback      = nullptr;
        params.abort_callback_data = nullptr;
        params.ctx_other           = nullptr;
    }
};

struct common_scoped_log_filter {
    ggml_log_callback callback  = nullptr;
    void *            user_data = nullptr;
    ggml_log_level    min_level;

    explicit common_scoped_log_filter(ggml_log_level level) : min_level(level) {
        llama_log_get(&callback, &user_data);
        llama_log_set(
            [](ggml_log_level message_level, const char * text, void * opaque) {
                const auto *         self = static_cast<const common_scoped_log_filter *>(opaque);
                const ggml_log_level effective =
                    message_level >= self->min_level ? message_level : GGML_LOG_LEVEL_DEBUG;
                self->callback(effective, text, self->user_data);
            },
            this);
    }

    ~common_scoped_log_filter() { llama_log_set(callback, user_data); }
};

struct common_joint_probe_owner {
    common_joint_component_owner owner;
    ggml_backend_dev_t           device         = nullptr;
    uint32_t                     selected_index = UINT32_MAX;
};

static bool common_joint_collect_component(llama_model *                           model,
                                           llama_context *                         context,
                                           const llama_moe_placement_report &      placement,
                                           bool                                    deduct_model,
                                           common_joint_component_measurement &    component,
                                           std::vector<common_joint_probe_owner> & probe_owners) {
    std::unordered_map<ggml_backend_dev_t, const llama_moe_placement_owner *> selected;
    const int32_t                                                             n_devices = llama_model_n_devices(model);
    if (n_devices < 0 || placement.owners.size() != static_cast<size_t>(n_devices)) {
        component.diagnostics.push_back("resolved placement owner count does not match the model device list");
        return false;
    }
    for (int32_t i = 0; i < n_devices; ++i) {
        selected.emplace(llama_model_get_device(model, i), &placement.owners[i]);
    }

    std::map<std::string, common_joint_probe_owner> by_owner;
    for (const auto & allocation : placement.model_allocations) {
        if (!allocation.bytes_available) {
            component.diagnostics.push_back("resolved model allocation size is unavailable");
            return false;
        }
        common_joint_probe_owner resolved;
        const bool               host =
            allocation.owner_canonical_id.empty() || allocation.resolved_class.find("host") != std::string::npos;
        if (host) {
            resolved.owner.canonical_id  = "host";
            resolved.owner.identity_kind = "process_host";
            resolved.owner.backend       = "CPU";
            resolved.owner.name          = "Host";
            resolved.owner.host          = true;
            resolved.device              = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        } else {
            const auto owner = std::find_if(placement.owners.begin(), placement.owners.end(), [&](const auto & value) {
                return value.canonical_id == allocation.owner_canonical_id;
            });
            if (owner == placement.owners.end()) {
                component.diagnostics.push_back("resolved model allocation owner is absent from physical placement");
                return false;
            }
            resolved.owner.canonical_id  = owner->canonical_id;
            resolved.owner.identity_kind = owner->identity_kind;
            resolved.owner.backend       = owner->backend;
            resolved.owner.name          = owner->name;
            resolved.selected_index      = owner->selected_index;
            resolved.device              = llama_model_get_device(model, owner->selected_index);
        }
        auto & aggregate = by_owner[resolved.owner.canonical_id];
        if (aggregate.owner.canonical_id.empty()) {
            aggregate = resolved;
        } else if (aggregate.device != resolved.device || aggregate.owner.host != resolved.owner.host) {
            component.diagnostics.push_back("canonical model owner maps to conflicting runtime devices");
            return false;
        }
        if (deduct_model) {
            if (!checked_add_size(component.shared_model_bytes_deduplicated, allocation.bytes)) {
                component.diagnostics.push_back("shared-model deduction overflow");
                return false;
            }
        } else if (!checked_add_size(aggregate.owner.memory.model, allocation.bytes)) {
            component.diagnostics.push_back("model allocation accounting overflow");
            return false;
        }
        if (!deduct_model && host && placement.uses_mmap && !allocation.current_allocation &&
            !checked_add_size(aggregate.owner.lower_bound_excluded_bytes, allocation.bytes)) {
            component.diagnostics.push_back("mapped-host lower-bound accounting overflow");
            return false;
        }
        aggregate.owner.provenance = allocation.provenance;
    }
    const llama_memory_breakdown breakdown = llama_get_memory_breakdown(context);
    for (const auto & [buft, memory] : breakdown) {
        common_joint_probe_owner resolved;
        if (ggml_backend_buft_is_host(buft)) {
            resolved.owner.canonical_id  = "host";
            resolved.owner.identity_kind = "process_host";
            resolved.owner.backend       = "CPU";
            resolved.owner.name          = "Host";
            resolved.owner.host          = true;
            resolved.device              = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        } else {
            ggml_backend_dev_t device = ggml_backend_buft_get_device(buft);
            const auto         found  = selected.find(device);
            if (device == nullptr || found == selected.end()) {
                component.diagnostics.push_back(
                    "memory buffer owner is absent from the resolved physical device mapping");
                return false;
            }
            const auto & owner           = *found->second;
            resolved.owner.canonical_id  = owner.canonical_id;
            resolved.owner.identity_kind = owner.identity_kind;
            resolved.owner.backend       = owner.backend;
            resolved.owner.name          = owner.name;
            resolved.device              = device;
            resolved.selected_index      = owner.selected_index;
        }
        auto & aggregate = by_owner[resolved.owner.canonical_id];
        if (aggregate.owner.canonical_id.empty()) {
            aggregate = resolved;
        } else if (aggregate.device != resolved.device || aggregate.owner.host != resolved.owner.host) {
            component.diagnostics.push_back("canonical owner identity maps to conflicting runtime devices");
            return false;
        }
        common_joint_memory addition = { 0, memory.context, memory.compute, 0, true };
        if (!checked_add_memory(aggregate.owner.memory, addition)) {
            component.diagnostics.push_back("component memory accounting overflow");
            return false;
        }
    }
    size_t         staging_bytes     = 0;
    const uint32_t staging_ctx_mask  = component.role == COMMON_JOINT_ROLE_MTP ? 2u : 1u;
    bool           staging_available = true;
    for (const auto & owner : placement.owners) {
        if ((owner.active_cache_context_mask & staging_ctx_mask) == 0) {
            continue;
        }
        if (!owner.mandatory_host_staging_available) {
            staging_available = false;
            continue;
        }
        const size_t required = component.role == COMMON_JOINT_ROLE_MTP ? owner.mandatory_host_staging_mtp_bytes :
                                                                          owner.mandatory_host_staging_default_bytes;
        if (!checked_add_size(staging_bytes, required)) {
            component.diagnostics.push_back("component host-staging accounting overflow");
            return false;
        }
    }
    if (!staging_available) {
        auto & host = by_owner["host"];
        if (host.owner.canonical_id.empty()) {
            host.owner.canonical_id  = "host";
            host.owner.identity_kind = "process_host";
            host.owner.backend       = "CPU";
            host.owner.name          = "Host";
            host.owner.host          = true;
            host.device              = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
        host.owner.memory.staging_available = false;
        host.owner.provenance               = "backend_memory_breakdown_includes_unclassified_host_staging";
    } else if (staging_bytes != 0) {
        auto & host = by_owner["host"];
        if (host.owner.canonical_id.empty() || host.owner.memory.context < staging_bytes) {
            component.diagnostics.push_back("host staging exceeds the backend context-memory breakdown");
            return false;
        }
        host.owner.memory.context -= staging_bytes;
        host.owner.memory.staging = staging_bytes;
        host.owner.provenance     = "placement_manifest_mandatory_host_staging";
    }
    for (auto & [id, owner] : by_owner) {
        component.owners.push_back(owner.owner);
        probe_owners.push_back(std::move(owner));
    }
    return true;
}

static common_json common_joint_memory_json(const common_joint_memory & memory) {
    common_json result      = common_json::object();
    result["model_bytes"]   = memory.model;
    result["context_bytes"] = memory.context;
    result["compute_bytes"] = memory.compute;
    result["staging_bytes"] = memory.staging_available ? common_json_value(memory.staging) : common_json_value(nullptr);
    size_t total            = 0;
    result["total_bytes"]   = memory_total(memory, total) ? common_json_value(total) : common_json_value(nullptr);
    return result;
}

common_joint_measurement common_measure_joint_configuration(const common_joint_measurement_request & request) {
    common_joint_measurement result;
    if (request.target.role != COMMON_JOINT_ROLE_TARGET || request.target.path_model.empty()) {
        result.diagnostics.push_back("joint measurement requires a non-empty target model");
        return result;
    }

    try {
        common_scoped_log_filter  log_filter(request.log_level);
        common_owned_model_params target_params(request.target.mparams);
        llama_model_ptr           target_model(
            llama_model_load_from_file(request.target.path_model.c_str(), target_params.params));
        if (!target_model) {
            result.diagnostics.push_back("failed to load required target metadata probe");
            return result;
        }
        common_owned_context_params target_context_params(request.target.cparams);
        llama_context_params &      target_cparams = target_context_params.params;
        llama_context_ptr           target_context(llama_init_from_model(target_model.get(), target_cparams));
        if (!target_context) {
            result.diagnostics.push_back("failed to create required target context probe");
            return result;
        }
        target_cparams.n_ctx = llama_n_ctx(target_context.get());

        std::vector<common_joint_probe_owner> all_probe_owners;
        std::vector<ggml_backend_dev_t>       target_devices;
        const int32_t                         n_target_devices = llama_model_n_devices(target_model.get());
        for (int32_t i = 0; i < n_target_devices; ++i) {
            target_devices.push_back(llama_model_get_device(target_model.get(), i));
        }
        const auto                         target_placement = llama_model_moe_placement(target_model.get());
        std::map<std::string, std::string> target_storage_owners;
        for (const auto & owner : target_placement.owners) {
            target_storage_owners.emplace(owner.canonical_id, owner.identity_kind);
        }
        for (const auto & allocation : target_placement.model_allocations) {
            if (!allocation.owner_canonical_id.empty()) {
                target_storage_owners.emplace(allocation.owner_canonical_id, allocation.owner_identity_kind);
            }
            if (allocation.owner_canonical_id.empty() && allocation.resolved_class.find("host") != std::string::npos) {
                target_storage_owners.emplace("host", "process_host");
            }
        }

        auto measure_component = [&](const common_joint_component_request & spec, llama_model * model,
                                     llama_context * context, const llama_model_params & effective_mparams,
                                     const llama_context_params & effective_cparams, bool deduct_model) {
            common_joint_component_measurement component;
            component.role                    = spec.role;
            component.sharing                 = spec.sharing;
            component.required                = spec.required;
            const auto placement              = llama_model_moe_placement(model);
            component.model_identity          = placement.model_identity;
            component.model_identity_kind     = placement.model_identity_kind;
            component.placement_id            = placement.placement_id;
            component.placement_identity_kind = placement.placement_identity_kind;
            bool sharing_resolved             = true;
            for (const auto & shared : placement.shared_tensors) {
                if (!checked_add_size(component.shared_tensor_payload_bytes, shared.tensor_bytes)) {
                    component.diagnostics.push_back("shared tensor payload accounting overflow");
                    component.completeness = COMMON_JOINT_COMPLETENESS_ERROR;
                    return component;
                }
                std::string owner_id       = shared.owner_canonical_id;
                std::string owner_kind     = shared.owner_identity_kind;
                std::string provenance     = shared.storage_provenance;
                bool        owner_resolved = shared.resolved_storage_available && !owner_id.empty();
                if (!owner_resolved && spec.sharing == COMMON_JOINT_SHARING_BORROW_TARGET &&
                    shared.storage_relation == "borrowed_model_shared" &&
                    shared.resolved_class.find("host") != std::string::npos &&
                    target_storage_owners.count("host") != 0) {
                    owner_id   = "host";
                    owner_kind = "process_host";
                    provenance += "+joint_process_host_normalization";
                    owner_resolved = true;
                }
                component.sharing_records.push_back({
                    shared.name,
                    owner_id,
                    owner_kind,
                    shared.tensor_bytes,
                    shared.storage_relation,
                    provenance,
                });
                if (!owner_resolved) {
                    sharing_resolved = false;
                    component.diagnostics.push_back("shared tensor ownership is unresolved: " + shared.name);
                } else if (spec.sharing == COMMON_JOINT_SHARING_BORROW_TARGET) {
                    const auto target_owner = target_storage_owners.find(owner_id);
                    if (target_owner == target_storage_owners.end() || target_owner->second != owner_kind) {
                        sharing_resolved = false;
                        component.diagnostics.push_back("shared tensor owner does not match target storage: " +
                                                        shared.name);
                    }
                }
            }
            if (spec.sharing == COMMON_JOINT_SHARING_TARGET_MODEL) {
                component.sharing_provenance = "same_model_model_bytes_deduplicated";
            } else if (!placement.shared_tensors.empty()) {
                component.sharing_provenance = "resolved_borrowed_aliases_excluded_from_unique_model_buffers";
            } else {
                component.sharing_provenance = "no_resolved_model_sharing";
            }
            if (!common_joint_collect_component(model, context, placement, deduct_model, component, all_probe_owners)) {
                component.completeness = COMMON_JOINT_COMPLETENESS_ERROR;
                return component;
            }
            const bool mapped_host_upper =
                std::any_of(component.owners.begin(), component.owners.end(),
                            [](const auto & owner) { return owner.lower_bound_excluded_bytes != 0; });
            component.bound             = !sharing_resolved ? COMMON_JOINT_BOUND_UNKNOWN :
                                          mapped_host_upper ? COMMON_JOINT_BOUND_UPPER_ESTIMATE :
                                                              COMMON_JOINT_BOUND_EXACT_ESTIMATE;
            component.memory_provenance = !sharing_resolved ?
                                              "no_alloc_estimate_with_unresolved_borrowed_alias_ownership" :
                                          mapped_host_upper ? "no_alloc_estimate_with_mapped_host_model_upper" :
                                                              "no_alloc_backend_allocation_size_estimate";
            const bool sampler_identity_resolved =
                effective_cparams.n_samplers == 0 || !spec.sampler_configuration_id.empty();
            if (!sampler_identity_resolved) {
                component.diagnostics.push_back(
                    "sampler-backed context requires a stable sampler configuration identity");
            }
            if (sampler_identity_resolved) {
                const common_json configuration = common_moe_configuration_record(
                    placement, effective_mparams, effective_cparams, request.runtime_params,
                    spec.sampler_configuration_id, common_joint_role_name(spec.role));
                component.configuration_record = configuration.dump();
                component.configuration_id =
                    hash_sha256_hex(component.configuration_record.data(), component.configuration_record.size());
            }
            component.completeness = sharing_resolved && sampler_identity_resolved ?
                                         COMMON_JOINT_COMPLETENESS_COMPLETE :
                                         COMMON_JOINT_COMPLETENESS_INCOMPLETE;
            return component;
        };

        result.components.push_back(measure_component(request.target, target_model.get(), target_context.get(),
                                                      target_params.params, target_cparams, false));
        if (result.components.back().completeness != COMMON_JOINT_COMPLETENESS_COMPLETE) {
            result.completeness = result.components.back().completeness;
            result.diagnostics.push_back("required target measurement is not complete");
            for (const auto & diagnostic : result.components.back().diagnostics) {
                result.diagnostics.push_back(std::string("target: ") + diagnostic);
            }
            return result;
        }

        for (const auto & extra : request.extras) {
            common_joint_component_measurement component;
            component.role     = extra.role;
            component.sharing  = extra.sharing;
            component.required = extra.required;
            try {
                common_owned_context_params extra_context_params(extra.cparams);
                llama_context_params &      extra_cparams = extra_context_params.params;
                extra_cparams.n_ctx                       = target_cparams.n_ctx;
                extra_cparams.ctx_other                   = target_context.get();
                if (extra.role == COMMON_JOINT_ROLE_MTP) {
                    extra_cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                } else if (extra.role == COMMON_JOINT_ROLE_DRAFT) {
                    extra_cparams.ctx_type = LLAMA_CONTEXT_TYPE_DRAFT;
                }

                if (extra.sharing == COMMON_JOINT_SHARING_TARGET_MODEL) {
                    llama_context_ptr extra_context(llama_init_from_model(target_model.get(), extra_cparams));
                    if (!extra_context) {
                        throw std::runtime_error("failed to create shared target-model context probe");
                    }
                    component = measure_component(extra, target_model.get(), extra_context.get(), target_params.params,
                                                  extra_cparams, true);
                } else {
                    if (extra.path_model.empty()) {
                        throw std::runtime_error("separate extra model path is empty");
                    }
                    common_owned_model_params extra_params(extra.mparams);
                    if (extra.sharing == COMMON_JOINT_SHARING_BORROW_TARGET) {
                        extra_params.params.model_shared = target_model.get();
                    }
                    llama_model_ptr extra_model(
                        llama_model_load_from_file(extra.path_model.c_str(), extra_params.params));
                    if (!extra_model) {
                        throw std::runtime_error("failed to load extra model metadata probe");
                    }
                    llama_context_ptr extra_context(llama_init_from_model(extra_model.get(), extra_cparams));
                    if (!extra_context) {
                        throw std::runtime_error("failed to create extra context probe");
                    }
                    component = measure_component(extra, extra_model.get(), extra_context.get(), extra_params.params,
                                                  extra_cparams, false);
                }
            } catch (const std::exception & error) {
                component.completeness =
                    extra.required ? COMMON_JOINT_COMPLETENESS_ERROR : COMMON_JOINT_COMPLETENESS_INCOMPLETE;
                component.diagnostics.push_back(error.what());
            }
            result.components.push_back(std::move(component));
        }

        result.completeness = COMMON_JOINT_COMPLETENESS_COMPLETE;
        for (const auto & component : result.components) {
            if (component.completeness == COMMON_JOINT_COMPLETENESS_COMPLETE) {
                continue;
            }
            if (component.required && component.completeness == COMMON_JOINT_COMPLETENESS_ERROR) {
                result.completeness = COMMON_JOINT_COMPLETENESS_ERROR;
            } else if (result.completeness == COMMON_JOINT_COMPLETENESS_COMPLETE) {
                result.completeness = COMMON_JOINT_COMPLETENESS_INCOMPLETE;
            }
            for (const auto & diagnostic : component.diagnostics) {
                result.diagnostics.push_back(std::string(common_joint_role_name(component.role)) + ": " + diagnostic);
            }
        }

        std::map<std::string, common_joint_owner_measurement> owners;
        std::map<std::string, ggml_backend_dev_t>             owner_devices;
        std::map<std::string, uint32_t>                       target_selected_indices;
        std::map<std::string, size_t>                         target_compute;
        std::map<std::string, size_t>                         shared_mtp_compute;
        for (size_t ci = 0; ci < result.components.size(); ++ci) {
            for (const auto & owner : result.components[ci].owners) {
                auto & aggregate = owners[owner.canonical_id];
                if (aggregate.canonical_id.empty()) {
                    aggregate.canonical_id  = owner.canonical_id;
                    aggregate.identity_kind = owner.identity_kind;
                    aggregate.backend       = owner.backend;
                    aggregate.name          = owner.name;
                    aggregate.host          = owner.host;
                } else if (aggregate.identity_kind != owner.identity_kind || aggregate.host != owner.host) {
                    result.completeness = COMMON_JOINT_COMPLETENESS_ERROR;
                    result.diagnostics.push_back("conflicting physical owner identity across components");
                    continue;
                }
                if (!checked_add_memory(aggregate.memory, owner.memory)) {
                    result.completeness = COMMON_JOINT_COMPLETENESS_ERROR;
                    result.diagnostics.push_back("joint owner memory accounting overflow");
                }
                if (!checked_add_size(aggregate.lower_bound_excluded_bytes, owner.lower_bound_excluded_bytes)) {
                    result.completeness = COMMON_JOINT_COMPLETENESS_ERROR;
                    result.diagnostics.push_back("joint lower-bound exclusion overflow");
                }
                if (result.components[ci].role == COMMON_JOINT_ROLE_TARGET) {
                    target_compute[owner.canonical_id] = owner.memory.compute;
                } else if (result.components[ci].role == COMMON_JOINT_ROLE_MTP &&
                           result.components[ci].sharing == COMMON_JOINT_SHARING_TARGET_MODEL) {
                    shared_mtp_compute[owner.canonical_id] = owner.memory.compute;
                }
            }
        }
        for (const auto & probe_owner : all_probe_owners) {
            if (!probe_owner.owner.host) {
                auto [it, inserted] = owner_devices.emplace(probe_owner.owner.canonical_id, probe_owner.device);
                if (!inserted && it->second != probe_owner.device) {
                    result.completeness = COMMON_JOINT_COMPLETENESS_ERROR;
                    result.diagnostics.push_back("physical owner maps to conflicting backend device handles");
                }
            }
        }
        if (!result.components.empty()) {
            for (const auto & probe_owner : all_probe_owners) {
                if (std::find(target_devices.begin(), target_devices.end(), probe_owner.device) ==
                    target_devices.end()) {
                    continue;
                }
                target_selected_indices.emplace(probe_owner.owner.canonical_id, probe_owner.selected_index);
            }
        }

        std::map<std::string, common_joint_device_limit> explicit_limits;
        for (const auto & limit : request.device_limits) {
            if (limit.canonical_id.empty() || !explicit_limits.emplace(limit.canonical_id, limit).second) {
                result.completeness = COMMON_JOINT_COMPLETENESS_ERROR;
                result.diagnostics.push_back("joint device limits contain an empty or duplicate canonical owner");
            }
        }
        const bool possible_shared_workspace =
            request.target.cparams.phase_aware_workspace &&
            std::any_of(request.extras.begin(), request.extras.end(), [](const auto & extra) {
                return extra.role == COMMON_JOINT_ROLE_MTP && extra.sharing == COMMON_JOINT_SHARING_TARGET_MODEL;
            });

        bool any_device           = false;
        bool any_unknown_capacity = false;
        bool any_exceeded         = false;
        for (auto & [id, owner] : owners) {
            size_t upper = 0;
            if (!memory_total(owner.memory, upper)) {
                result.completeness = COMMON_JOINT_COMPLETENESS_ERROR;
                result.diagnostics.push_back("joint memory total overflow");
            }
            owner.required_upper_bytes = upper;
            owner.required_lower_bytes = upper - owner.lower_bound_excluded_bytes;
            owner.bound                = owner.lower_bound_excluded_bytes == 0 ? COMMON_JOINT_BOUND_EXACT_ESTIMATE :
                                                                                 COMMON_JOINT_BOUND_UPPER_ESTIMATE;
            owner.memory_provenance    = owner.lower_bound_excluded_bytes == 0 ?
                                             "no_alloc_backend_allocation_size_estimate" :
                                             "no_alloc_mapped_host_model_lower_excluded_upper_included";
            if (possible_shared_workspace) {
                const size_t target_value       = target_compute[id];
                const size_t mtp_value          = shared_mtp_compute[id];
                const size_t possible_deduction = std::min(target_value, mtp_value);
                owner.required_lower_bytes -= possible_deduction;
                if (possible_deduction > 0) {
                    owner.bound             = COMMON_JOINT_BOUND_UPPER_ESTIMATE;
                    owner.memory_provenance = "no_alloc_conservative_independent_workspace_upper";
                }
            }
            any_device = true;
            if (!owner.host) {
                const auto selected = target_selected_indices.find(id);
                owner.margin_bytes  = selected != target_selected_indices.end() &&
                                              selected->second < request.target_device_margins.size() ?
                                          request.target_device_margins[selected->second] :
                                          request.default_device_margin;
            }
            const auto explicit_limit = explicit_limits.find(id);
            const auto device         = owner_devices.find(id);
            if (explicit_limit != explicit_limits.end()) {
                owner.capacity_available  = explicit_limit->second.capacity_available;
                owner.capacity_bytes      = explicit_limit->second.capacity_bytes;
                owner.total_bytes         = explicit_limit->second.capacity_bytes;
                owner.margin_bytes        = explicit_limit->second.margin_bytes;
                owner.capacity_provenance = explicit_limit->second.provenance;
            } else if (owner.host && request.host_capacity_bytes.has_value()) {
                owner.capacity_available  = true;
                owner.capacity_bytes      = *request.host_capacity_bytes;
                owner.total_bytes         = *request.host_capacity_bytes;
                owner.margin_bytes        = request.host_margin_bytes;
                owner.capacity_provenance = request.host_capacity_provenance;
            } else if (request.observe_device_capacity) {
                ggml_backend_dev_t capacity_device = nullptr;
                if (owner.host) {
                    capacity_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                } else if (device != owner_devices.end()) {
                    capacity_device = device->second;
                }
                if (capacity_device != nullptr) {
                    size_t free  = 0;
                    size_t total = 0;
                    ggml_backend_dev_memory(capacity_device, &free, &total);
                    if (free != 0 || total != 0) {
                        owner.capacity_available  = true;
                        owner.capacity_bytes      = free;
                        owner.total_bytes         = total;
                        owner.capacity_provenance = "backend_current_free_observation";
                    }
                }
            }
            if (owner.required_upper_bytes > static_cast<size_t>(INT64_MAX) ||
                owner.required_lower_bytes > static_cast<size_t>(INT64_MAX) ||
                owner.capacity_bytes > static_cast<size_t>(INT64_MAX) ||
                owner.margin_bytes > static_cast<size_t>(INT64_MAX)) {
                result.completeness = COMMON_JOINT_COMPLETENESS_ERROR;
                result.diagnostics.push_back("capacity arithmetic exceeds signed reporting range");
            } else if (owner.capacity_available) {
                const size_t usable =
                    owner.margin_bytes <= owner.capacity_bytes ? owner.capacity_bytes - owner.margin_bytes : 0;
                owner.slack_bytes        = owner.required_upper_bytes <= usable ?
                                               static_cast<int64_t>(usable - owner.required_upper_bytes) :
                                               -static_cast<int64_t>(owner.required_upper_bytes - usable);
                const bool lower_exceeds = owner.required_lower_bytes > usable;
                any_exceeded             = any_exceeded || lower_exceeds;
                any_unknown_capacity = any_unknown_capacity || (!lower_exceeds && owner.required_upper_bytes > usable);
            } else {
                any_unknown_capacity      = true;
                owner.capacity_provenance = "unavailable";
            }
            result.owners.push_back(std::move(owner));
        }
        if (any_exceeded) {
            result.capacity = COMMON_JOINT_CAPACITY_EXCEEDS_LIMITS;
        } else if (!any_device || any_unknown_capacity) {
            result.capacity = COMMON_JOINT_CAPACITY_UNKNOWN;
        } else {
            result.capacity = COMMON_JOINT_CAPACITY_WITHIN_LIMITS;
        }

        bool        all_components_identified = result.completeness == COMMON_JOINT_COMPLETENESS_COMPLETE;
        common_json canonical                 = common_json::object();
        common_json components                = common_json::array();
        for (const auto & component : result.components) {
            if (component.configuration_record.empty()) {
                all_components_identified = false;
                continue;
            }
            common_json entry      = common_json::object();
            entry["configuration"] = common_json::parse(component.configuration_record);
            entry["required"]      = component.required;
            entry["role"]          = common_joint_role_name(component.role);
            entry["sharing"]       = common_joint_sharing_name(component.sharing);
            components.push_back(std::move(entry));
        }
        common_json limits = common_json::array();
        for (const auto & owner : result.owners) {
            common_json limit          = common_json({
                { "canonical_id", owner.canonical_id },
                { "margin_bytes", owner.margin_bytes },
            });
            const auto  explicit_limit = explicit_limits.find(owner.canonical_id);
            if (explicit_limit != explicit_limits.end()) {
                limit["capacity_bytes"]      = explicit_limit->second.capacity_available ?
                                                   common_json_value(explicit_limit->second.capacity_bytes) :
                                                   common_json_value(nullptr);
                limit["capacity_provenance"] = explicit_limit->second.provenance;
            } else if (owner.host && request.host_capacity_bytes.has_value()) {
                limit["capacity_bytes"]      = *request.host_capacity_bytes;
                limit["capacity_provenance"] = request.host_capacity_provenance;
            }
            limits.push_back(std::move(limit));
        }
        canonical["components"]     = std::move(components);
        canonical["device_limits"]  = std::move(limits);
        canonical["schema_version"] = result.schema_version;
        if (all_components_identified) {
            result.configuration_record = canonical.dump();
            result.configuration_id =
                hash_sha256_hex(result.configuration_record.data(), result.configuration_record.size());
        } else {
            result.diagnostics.push_back("joint configuration identity unavailable because a component was unresolved");
        }
        result.admission_qualified = result.completeness == COMMON_JOINT_COMPLETENESS_COMPLETE &&
                                     result.capacity == COMMON_JOINT_CAPACITY_WITHIN_LIMITS &&
                                     !result.configuration_id.empty();
    } catch (const std::exception & error) {
        result.completeness        = COMMON_JOINT_COMPLETENESS_ERROR;
        result.capacity            = COMMON_JOINT_CAPACITY_UNKNOWN;
        result.admission_qualified = false;
        result.diagnostics.push_back(error.what());
    }
    return result;
}

struct common_joint_sampler_bundle {
    // Model-dependent sampler implementations may retain vocabulary pointers.
    // Keep their allocation-only model alive until the measurement has cloned
    // and destroyed every sampler attachment.
    llama_model_ptr                          model;
    std::vector<common_sampler_ptr>        common;
    std::vector<llama_sampler_ptr>         raw;
    std::vector<llama_sampler_seq_config>  configs;
};

static std::string common_joint_float(float value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::hexfloat << value;
    return out.str();
}

static std::string common_joint_sampling_configuration_id(
        const common_params_sampling & params,
        const char *                   kind,
        uint32_t                       n_seq) {
    common_json record = common_json::object();
    record["kind"]                    = kind;
    record["n_seq"]                   = n_seq;
    record["seed"]                    = params.seed;
    record["n_prev"]                  = params.n_prev;
    record["n_probs"]                 = params.n_probs;
    record["min_keep"]                = params.min_keep;
    record["top_k"]                   = params.top_k;
    record["top_p"]                   = common_joint_float(params.top_p);
    record["min_p"]                   = common_joint_float(params.min_p);
    record["xtc_probability"]         = common_joint_float(params.xtc_probability);
    record["xtc_threshold"]           = common_joint_float(params.xtc_threshold);
    record["typ_p"]                   = common_joint_float(params.typ_p);
    record["temp"]                    = common_joint_float(params.temp);
    record["dynatemp_range"]          = common_joint_float(params.dynatemp_range);
    record["dynatemp_exponent"]       = common_joint_float(params.dynatemp_exponent);
    record["penalty_last_n"]          = params.penalty_last_n;
    record["penalty_repeat"]          = common_joint_float(params.penalty_repeat);
    record["penalty_freq"]            = common_joint_float(params.penalty_freq);
    record["penalty_present"]         = common_joint_float(params.penalty_present);
    record["dry_multiplier"]          = common_joint_float(params.dry_multiplier);
    record["dry_base"]                = common_joint_float(params.dry_base);
    record["dry_allowed_length"]      = params.dry_allowed_length;
    record["dry_penalty_last_n"]      = params.dry_penalty_last_n;
    record["adaptive_target"]         = common_joint_float(params.adaptive_target);
    record["adaptive_decay"]          = common_joint_float(params.adaptive_decay);
    record["mirostat"]                = params.mirostat;
    record["top_n_sigma"]             = common_joint_float(params.top_n_sigma);
    record["mirostat_tau"]            = common_joint_float(params.mirostat_tau);
    record["mirostat_eta"]            = common_joint_float(params.mirostat_eta);
    record["ignore_eos"]              = params.ignore_eos;
    record["no_perf"]                 = params.no_perf;
    record["timing_per_token"]        = params.timing_per_token;
    record["user_sampling_config"]    = params.user_sampling_config;
    record["dry_sequence_breakers"]   = params.dry_sequence_breakers;
    record["grammar_type"]            = static_cast<int32_t>(params.grammar.type);
    record["grammar"]                 = params.grammar.grammar;
    record["grammar_lazy"]            = params.grammar_lazy;
    record["generation_prompt"]       = params.generation_prompt;
    record["reasoning_budget_tokens"] = params.reasoning_budget_tokens;
    record["reasoning_budget_start"]  = params.reasoning_budget_start;
    common_json reasoning_budget_end = common_json::array();
    for (const auto & sequence : params.reasoning_budget_end) {
        reasoning_budget_end.push_back(sequence);
    }
    record["reasoning_budget_end"]    = std::move(reasoning_budget_end);
    record["reasoning_budget_forced"] = params.reasoning_budget_forced;
    record["reasoning_budget_message"] = params.reasoning_budget_message;
    record["reasoning_control"]        = params.reasoning_control;
    record["backend_sampling"]         = params.backend_sampling;

    common_json samplers = common_json::array();
    for (const auto sampler : params.samplers) {
        samplers.push_back(static_cast<int32_t>(sampler));
    }
    record["samplers"] = std::move(samplers);

    common_json triggers = common_json::array();
    for (const auto & trigger : params.grammar_triggers) {
        triggers.push_back({
            { "type",  static_cast<int32_t>(trigger.type) },
            { "value", trigger.value                       },
            { "token", trigger.token                       },
        });
    }
    record["grammar_triggers"] = std::move(triggers);

    common_json preserved = common_json::array();
    for (const auto token : params.preserved_tokens) {
        preserved.push_back(token);
    }
    record["preserved_tokens"] = std::move(preserved);

    const auto encode_biases = [](const std::vector<llama_logit_bias> & biases) {
        common_json result = common_json::array();
        for (const auto & bias : biases) {
            result.push_back({
                { "token", bias.token                    },
                { "bias",  common_joint_float(bias.bias) },
            });
        }
        return result;
    };
    record["logit_bias"]     = encode_biases(params.logit_bias);
    record["logit_bias_eog"] = encode_biases(params.logit_bias_eog);

    const std::string serialized = record.dump();
    return hash_sha256_hex(serialized.data(), serialized.size());
}

static bool common_joint_attach_target_samplers(
        common_params &                         params,
        common_joint_component_request &        target,
        common_joint_sampler_bundle &           bundle,
        std::string &                           error) {
    if (!params.sampling.backend_sampling) {
        return true;
    }

    try {
        common_owned_model_params sampler_model_params(target.mparams);
        bundle.model.reset(llama_model_load_from_file(target.path_model.c_str(), sampler_model_params.params));
        if (!bundle.model) {
            error = "failed to load target sampler configuration probe";
            return false;
        }

        common_params_sampling_prepare(bundle.model.get(), params.sampling);
        bundle.common.reserve(target.cparams.n_seq_max);
        bundle.configs.reserve(target.cparams.n_seq_max);
        for (llama_seq_id seq = 0; seq < static_cast<llama_seq_id>(target.cparams.n_seq_max); ++seq) {
            common_sampler_ptr sampler(common_sampler_init(bundle.model.get(), params.sampling));
            if (!sampler) {
                error = "failed to initialize target backend sampler probe";
                return false;
            }
            bundle.configs.push_back({ seq, common_sampler_get(sampler.get()) });
            bundle.common.push_back(std::move(sampler));
        }
        if (!params.sampling.backend_sampling) {
            bundle.configs.clear();
            bundle.common.clear();
            bundle.model.reset();
            return true;
        }

        target.cparams.samplers   = bundle.configs.data();
        target.cparams.n_samplers = bundle.configs.size();
        target.sampler_configuration_id =
            common_joint_sampling_configuration_id(params.sampling, "target_common_sampler", target.cparams.n_seq_max);
        return true;
    } catch (const std::exception & exception) {
        error = std::string("failed to prepare target backend sampler probe: ") + exception.what();
        return false;
    }
}

static bool common_joint_attach_draft_samplers(
        const common_params &                   params,
        common_joint_component_request &        extra,
        common_joint_sampler_bundle &           bundle,
        std::string &                           error) {
    if (!params.speculative.draft.backend_sampling) {
        return true;
    }

    const auto has_type = [&](common_speculative_type type) {
        return std::find(params.speculative.types.begin(), params.speculative.types.end(), type) !=
               params.speculative.types.end();
    };
    if (!has_type(COMMON_SPECULATIVE_TYPE_DRAFT_MTP)) {
        if (has_type(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3) ||
            has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) ||
            has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)) {
            error = "strict joint measurement does not yet represent this draft backend sampler";
            return false;
        }
        // The simple draft implementation samples on the CPU and does not
        // attach a sampler to its backend context.
        return true;
    }

    try {
        bundle.raw.reserve(extra.cparams.n_seq_max);
        bundle.configs.reserve(extra.cparams.n_seq_max);
        for (llama_seq_id seq = 0; seq < static_cast<llama_seq_id>(extra.cparams.n_seq_max); ++seq) {
            llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
            if (!chain) {
                error = "failed to initialize draft backend sampler probe";
                return false;
            }
            llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(10));
            bundle.configs.push_back({ seq, chain.get() });
            bundle.raw.push_back(std::move(chain));
        }
        common_params_sampling draft_sampling;
        draft_sampling.no_perf          = false;
        draft_sampling.top_k            = 10;
        draft_sampling.samplers          = { COMMON_SAMPLER_TYPE_TOP_K };
        draft_sampling.backend_sampling = true;
        extra.cparams.samplers   = bundle.configs.data();
        extra.cparams.n_samplers = bundle.configs.size();
        extra.sampler_configuration_id =
            common_joint_sampling_configuration_id(draft_sampling, "draft_top_k_10", extra.cparams.n_seq_max);
        return true;
    } catch (const std::exception & exception) {
        error = std::string("failed to prepare draft backend sampler probe: ") + exception.what();
        return false;
    }
}

common_joint_measurement common_measure_joint_configuration(const common_params & params,
                                                            ggml_log_level        log_level,
                                                            bool                  observe_device_capacity) {
    common_params                    target_params = params;
    common_joint_measurement_request request;
    request.target.path_model       = params.model.path;
    request.target.mparams          = common_model_params_to_llama(target_params);
    request.target.cparams          = common_context_params_to_llama(target_params);
    request.target.role             = COMMON_JOINT_ROLE_TARGET;
    request.target.required         = true;
    request.target_device_margins   = params.fit_params_target;
    request.default_device_margin   = params.fit_params_target.empty() ? 0 : params.fit_params_target.front();
    request.log_level               = log_level;
    request.observe_device_capacity = observe_device_capacity;
    request.runtime_params          = &target_params;

    common_joint_sampler_bundle target_samplers;
    std::string                 sampler_error;
    if (!common_joint_attach_target_samplers(target_params, request.target, target_samplers, sampler_error)) {
        common_joint_measurement result;
        result.diagnostics.push_back(std::move(sampler_error));
        return result;
    }

    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp  = std::find(params.speculative.types.begin(), params.speculative.types.end(),
                                     COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
    std::optional<common_params> draft_params;
    common_joint_sampler_bundle draft_samplers;
    if (has_draft || spec_mtp) {
        draft_params.emplace(common_base_params_to_speculative(target_params));
        common_joint_component_request extra;
        extra.path_model       = has_draft ? draft_params->model.path : params.model.path;
        extra.mparams          = common_model_params_to_llama(*draft_params);
        extra.cparams          = common_context_params_to_llama(*draft_params);
        extra.cparams.n_rs_seq = 0;
        extra.role             = spec_mtp ? COMMON_JOINT_ROLE_MTP : COMMON_JOINT_ROLE_DRAFT;
        extra.sharing          = has_draft ? COMMON_JOINT_SHARING_BORROW_TARGET : COMMON_JOINT_SHARING_TARGET_MODEL;
        extra.required         = true;
        if (!common_joint_attach_draft_samplers(target_params, extra, draft_samplers, sampler_error)) {
            common_joint_measurement result;
            result.diagnostics.push_back(std::move(sampler_error));
            return result;
        }
        request.extras.push_back(std::move(extra));
    }
    return common_measure_joint_configuration(request);
}

std::string common_joint_measurement_json(const common_joint_measurement & measurement) {
    common_json root         = common_json::object();
    root["schema_version"]   = measurement.schema_version;
    root["report_kind"]      = "moe_joint_measurement";
    root["configuration_id"] = measurement.configuration_id.empty() ? common_json_value(nullptr) :
                                                                      common_json_value(measurement.configuration_id);
    if (measurement.configuration_record.empty()) {
        root["configuration"] = nullptr;
    } else {
        root["configuration"] = common_json::parse(measurement.configuration_record);
    }
    root["measurement_completeness"] = common_joint_completeness_name(measurement.completeness);
    root["capacity_assessment"]      = common_joint_capacity_name(measurement.capacity);
    root["admission_qualified"]      = measurement.admission_qualified;
    root["diagnostics"]              = measurement.diagnostics;
    common_json components           = common_json::array();
    for (const auto & component : measurement.components) {
        common_json item                 = common_json::object();
        item["role"]                     = common_joint_role_name(component.role);
        item["sharing"]                  = common_joint_sharing_name(component.sharing);
        item["required"]                 = component.required;
        item["measurement_completeness"] = common_joint_completeness_name(component.completeness);
        item["configuration_id"] = component.configuration_id.empty() ? common_json_value(nullptr) :
                                                                        common_json_value(component.configuration_id);
        if (component.configuration_record.empty()) {
            item["configuration"] = nullptr;
        } else {
            item["configuration"] = common_json::parse(component.configuration_record);
        }
        item["model_identity"] =
            component.model_identity.empty() ? common_json_value(nullptr) : common_json_value(component.model_identity);
        item["model_identity_kind"] = component.model_identity_kind.empty() ?
                                          common_json_value(nullptr) :
                                          common_json_value(component.model_identity_kind);
        item["placement_id"] =
            component.placement_id.empty() ? common_json_value(nullptr) : common_json_value(component.placement_id);
        item["placement_identity_kind"]         = component.placement_identity_kind.empty() ?
                                                      common_json_value(nullptr) :
                                                      common_json_value(component.placement_identity_kind);
        item["shared_model_bytes_deduplicated"] = component.shared_model_bytes_deduplicated;
        item["shared_tensor_payload_bytes"]     = component.shared_tensor_payload_bytes;
        item["sharing_provenance"]              = component.sharing_provenance;
        item["bound"]                           = common_joint_bound_name(component.bound);
        item["memory_provenance"]               = component.memory_provenance;
        item["diagnostics"]                     = component.diagnostics;
        common_json sharing                     = common_json::array();
        for (const auto & record : component.sharing_records) {
            sharing.push_back(common_json({
                { "tensor_name",          record.tensor_name          },
                { "owner_canonical_id",   record.owner_canonical_id   },
                { "owner_identity_kind",  record.owner_identity_kind  },
                { "tensor_payload_bytes", record.tensor_payload_bytes },
                { "relation",             record.relation             },
                { "provenance",           record.provenance           },
            }));
        }
        item["sharing_records"] = std::move(sharing);
        common_json owners      = common_json::array();
        for (const auto & owner : component.owners) {
            owners.push_back(common_json({
                { "canonical_id",               owner.canonical_id                     },
                { "identity_kind",              owner.identity_kind                    },
                { "backend",                    owner.backend                          },
                { "name",                       owner.name                             },
                { "host",                       owner.host                             },
                { "lower_bound_excluded_bytes", owner.lower_bound_excluded_bytes       },
                { "memory",                     common_joint_memory_json(owner.memory) },
                { "provenance",                 owner.provenance                       },
            }));
        }
        item["owners"] = std::move(owners);
        components.push_back(std::move(item));
    }
    root["components"] = std::move(components);
    common_json owners = common_json::array();
    for (const auto & owner : measurement.owners) {
        common_json item                   = common_json::object();
        item["canonical_id"]               = owner.canonical_id;
        item["identity_kind"]              = owner.identity_kind;
        item["backend"]                    = owner.backend;
        item["name"]                       = owner.name;
        item["host"]                       = owner.host;
        item["memory"]                     = common_joint_memory_json(owner.memory);
        item["required_lower_bytes"]       = owner.required_lower_bytes;
        item["required_upper_bytes"]       = owner.required_upper_bytes;
        item["lower_bound_excluded_bytes"] = owner.lower_bound_excluded_bytes;
        item["bound"]                      = common_joint_bound_name(owner.bound);
        item["memory_provenance"]          = owner.memory_provenance;
        item["capacity_bytes"] =
            owner.capacity_available ? common_json_value(owner.capacity_bytes) : common_json_value(nullptr);
        item["total_bytes"] =
            owner.capacity_available ? common_json_value(owner.total_bytes) : common_json_value(nullptr);
        item["margin_bytes"] = owner.margin_bytes;
        item["slack_bytes"] =
            owner.capacity_available ? common_json_value(owner.slack_bytes) : common_json_value(nullptr);
        item["capacity_provenance"] = owner.capacity_provenance;
        owners.push_back(std::move(item));
    }
    root["owners"] = std::move(owners);
    return root.dump();
}

std::string common_joint_measurement_human(const common_joint_measurement & measurement) {
    std::ostringstream out;
    out << "MoE joint measurement "
        << (measurement.configuration_id.empty() ? "unresolved" : measurement.configuration_id) << '\n'
        << "  status: " << common_joint_completeness_name(measurement.completeness)
        << "; capacity: " << common_joint_capacity_name(measurement.capacity)
        << "; admission: " << (measurement.admission_qualified ? "qualified" : "not-qualified") << '\n';
    for (const auto & component : measurement.components) {
        out << "  " << common_joint_role_name(component.role) << " required=" << (component.required ? "yes" : "no")
            << " sharing=" << common_joint_sharing_name(component.sharing)
            << " status=" << common_joint_completeness_name(component.completeness)
            << " placement=" << (component.placement_id.empty() ? "unknown" : component.placement_id) << '\n';
    }
    for (const auto & owner : measurement.owners) {
        size_t     total           = 0;
        const bool total_available = memory_total(owner.memory, total);
        out << "  owner " << owner.canonical_id << " bytes=" << (total_available ? std::to_string(total) : "overflow")
            << " model=" << owner.memory.model << " context=" << owner.memory.context
            << " compute=" << owner.memory.compute << " staging=" << owner.memory.staging
            << " bound=" << common_joint_bound_name(owner.bound) << " margin=" << owner.margin_bytes << " slack=";
        if (owner.capacity_available) {
            out << owner.slack_bytes;
        } else {
            out << "unknown";
        }
        out << '\n';
    }
    for (const auto & diagnostic : measurement.diagnostics) {
        out << "  diagnostic: " << diagnostic << '\n';
    }
    return out.str();
}

// this enum is only used in llama_params_fit_impl but needs to be defined outside of it to fix a Windows compilation issue
// enum to identify part of a layer for distributing its tensors:
enum common_layer_fraction_t {
    LAYER_FRACTION_NONE = 0, // nothing
    LAYER_FRACTION_ATTN = 1, // attention
    LAYER_FRACTION_UP   = 2, // attention + up
    LAYER_FRACTION_GATE = 3, // attention + up + gate
    LAYER_FRACTION_MOE  = 4, // everything but sparse MoE weights
};

class common_params_fit_exception : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

static std::vector<llama_device_memory_data> common_get_device_memory_data_impl(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc  = true;
    mparams_copy.load_mode = LLAMA_LOAD_MODE_NONE;

    llama_model * model = llama_model_load_from_file(path_model, mparams_copy);
    if (model == nullptr) {
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to load model");
    }

    llama_context * ctx = llama_init_from_model(model, *cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to create llama_context from model");
    }

    const size_t nd = llama_model_n_devices(model);
    std::vector<llama_device_memory_data> ret(nd + 1);

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    for (const auto & [buft, mb] : memory_breakdown) {
        if (ggml_backend_buft_is_host(buft)) {
            ret.back().mb.model   += mb.model;
            ret.back().mb.context += mb.context;
            ret.back().mb.compute += mb.compute;
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            continue;
        }
        for (size_t i = 0; i < nd; i++) {
            if (dev == llama_model_get_device(model, i)) {
                ret[i].mb.model   += mb.model;
                ret[i].mb.context += mb.context;
                ret[i].mb.compute += mb.compute;
                break;
            }
        }
    }

    {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error("no CPU backend found");
        }
        size_t free;
        size_t total;
        ggml_backend_dev_memory(cpu_dev, &free, &total);
        ret.back().free  = free;
        ret.back().total = total;
    }
    for (size_t i = 0; i < nd; i++) {
        ggml_backend_dev_t dev = llama_model_get_device(model, i);

        size_t free;
        size_t total;
        ggml_backend_dev_memory(dev, &free, &total);

        // Some non-GPU accelerator backends, such as BLAS, report 0/0 and rely on
        // the host-memory fallback. For GPU-like backends, keep 0/0 so --fit does
        // not assign anything to a device with an unknown memory budget.
        if (free == 0 && total == 0) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                LOG_WRN("%s: device %s did not report memory; --fit will not use it\n",
                        __func__, ggml_backend_dev_name(dev));
            } else {
                free  = ret.back().free;
                total = ret.back().total;
            }
        }
        ret[i].free  = free;
        ret[i].total = total;
    }

    devs.clear();
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devs.push_back(llama_model_get_device(model, i));
    }

    hp_ngl         = llama_model_n_layer(model);
    if (mparams->load_mtp) {
        hp_ngl    += llama_model_n_layer_nextn(model);
    }
    hp_n_ctx_train = llama_model_n_ctx_train(model);
    hp_n_expert    = llama_model_n_expert(model);

    common_memory_breakdown_print(ctx);

    llama_free(ctx);
    llama_model_free(model);
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);

    return ret;
}

common_device_memory_data_vec common_get_device_memory_data(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level) {
    std::vector<llama_device_memory_data> impl = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level);

    common_device_memory_data_vec ret(impl.size());
    for (size_t i = 0; i < impl.size(); i++) {
        ret[i].total   = impl[i].total;
        ret[i].free    = impl[i].free;
        ret[i].model   = impl[i].mb.model;
        ret[i].context = impl[i].mb.context;
        ret[i].compute = impl[i].mb.compute;
    }
    return ret;
}

static void common_params_fit_impl(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        float * tensor_split, struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins_s, uint32_t n_ctx_min, const common_fit_extra_model * extra, enum ggml_log_level log_level) {
    if (mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        throw common_params_fit_exception("llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort");
    }
    constexpr int64_t MiB = 1024*1024;
    typedef std::vector<llama_device_memory_data> dmds_t;
    const llama_model_params default_mparams = llama_model_default_params();

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    // size the context for all sequences, but keep minimums and alignment per KV stream
    const uint32_t n_seq_max  = std::max<uint32_t>(1, cparams->n_seq_max);
    const uint32_t n_streams  = cparams->kv_unified ? 1 : n_seq_max;
    const bool     n_ctx_auto = cparams->n_ctx == 0;

    dmds_t   dmds_extra;       // memory of the extra model, laid out on the devices of the main model
    uint32_t n_ctx_extra = 0;  // context that memory was measured at

    // the extra model competes for the same memory as the main model, add it to every measurement
    // its memory is measured again whenever the context it follows changes
    auto add_extra_memory = [&](dmds_t & dmds) {
        if (extra == nullptr) {
            return;
        }

        if (dmds_extra.empty() || n_ctx_extra != cparams->n_ctx) {
            std::vector<ggml_backend_dev_t> devs_extra;
            uint32_t ngl_extra = 0;
            uint32_t nct_extra = 0;
            uint32_t nex_extra = 0;

            extra->cparams->n_ctx = cparams->n_ctx;

            LOG_TRC("%s: getting device memory data for the extra model at a context size of %" PRIu32 ":\n",
                __func__, cparams->n_ctx);

            dmds_t measured;
            try {
                measured = common_get_device_memory_data_impl(
                    extra->path_model, extra->mparams, extra->cparams, devs_extra, ngl_extra, nct_extra, nex_extra, log_level);
            } catch (const std::runtime_error & e) {
                // the extra model is optional, fit the main model alone rather than giving up
                LOG_WRN("%s: failed to measure the memory of the extra model, fitting without it: %s\n", __func__, e.what());
                dmds_extra = dmds_t(devs.size() + 1);
                n_ctx_extra = cparams->n_ctx;
                return;
            }

            dmds_extra = dmds_t(devs.size() + 1);
            dmds_extra.back().mb = measured.back().mb;
            for (size_t je = 0; je < devs_extra.size(); je++) {
                for (size_t id = 0; id < devs.size(); id++) {
                    if (devs_extra[je] == devs[id]) {
                        dmds_extra[id].mb.model   += measured[je].mb.model;
                        dmds_extra[id].mb.context += measured[je].mb.context;
                        dmds_extra[id].mb.compute += measured[je].mb.compute;
                        break;
                    }
                }
            }
            if (extra->shares_model) {
                for (llama_device_memory_data & dmd : dmds_extra) {
                    dmd.mb.model = 0;
                }
            }

            n_ctx_extra = cparams->n_ctx;
        }

        for (size_t id = 0; id < dmds.size(); id++) {
            dmds[id].mb.model   += dmds_extra[id].mb.model;
            dmds[id].mb.context += dmds_extra[id].mb.context;
            dmds[id].mb.compute += dmds_extra[id].mb.compute;
        }
    };

    // step 1: get data for default parameters and check whether any changes are necessary in the first place

    LOG_TRC("%s: getting device memory data for initial parameters:\n", __func__);
    dmds_t dmds_full = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

    // saturate instead of overflowing, this also preserves the UINT32_MAX sentinel of n_ctx_min:
    const uint32_t n_ctx_max       = (uint32_t) std::min<uint64_t>(uint64_t(hp_nct)    * n_seq_max, UINT32_MAX);
    const uint32_t n_ctx_min_total = (uint32_t) std::min<uint64_t>(uint64_t(n_ctx_min) * n_streams, UINT32_MAX);

    // llama_context would use only hp_nct in total for n_ctx == 0, resolve the context before measuring anything else:
    if (n_ctx_auto) {
        cparams->n_ctx = n_ctx_max;
        if (n_seq_max > 1) {
            LOG_TRC("%s: context size unset -> using %" PRIu32 " for %" PRIu32 " sequences:\n",
                __func__, n_ctx_max, n_seq_max);
            dmds_full = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        }
    }
    add_extra_memory(dmds_full);

    const size_t nd = devs.size(); // number of devices

    std::vector<int64_t> margins; // this function uses int64_t rather than size_t for memory sizes to more conveniently handle deficits
    margins.reserve(nd);
    if (nd == 0) {
        margins.push_back(margins_s[0]);
    } else {
        for (size_t id = 0; id < nd; id++) {
            margins.push_back(margins_s[id]);
        }
    }

    std::vector<std::string> dev_names;
    {
        dev_names.reserve(nd);
        size_t max_length = 0;
        for (const auto & dev : devs) {
            std::string name = ggml_backend_dev_name(dev);
            name += " (";
            name += ggml_backend_dev_description(dev);
            name += ")";
            dev_names.push_back(name);
            max_length = std::max(max_length, name.length());
        }
        for (std::string & dn : dev_names) {
            dn.insert(dn.end(), max_length - dn.length(), ' ');
        }
    }

    int64_t sum_free            = 0;
    int64_t sum_projected_free  = 0;
    int64_t sum_projected_used  = 0;
    int64_t sum_projected_model = 0;
    std::vector<int64_t> projected_free_per_device;
    projected_free_per_device.reserve(nd);

    if (nd == 0) {
        sum_projected_used = dmds_full.back().mb.total();
        sum_free           = dmds_full.back().total;
        sum_projected_free = sum_free - sum_projected_used;
        LOG_TRC("%s: projected to use %" PRId64 " MiB of host memory vs. %" PRId64 " MiB of total host memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (sum_projected_free >= margins[0]) {
            LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of system memory, no changes needed\n",
                __func__, sum_projected_free/MiB, margins[0]/MiB);
            return;
        }
    } else {
        if (nd > 1) {
            LOG_TRC("%s: projected memory use with initial parameters [MiB]:\n", __func__);
        }
        for (size_t id = 0; id < nd; id++) {
            const llama_device_memory_data & dmd = dmds_full[id];

            const int64_t projected_used = dmd.mb.total();
            const int64_t projected_free = dmd.free - projected_used;
            projected_free_per_device.push_back(projected_free);

            sum_free            += dmd.free;
            sum_projected_used  += projected_used;
            sum_projected_free  += projected_free;
            sum_projected_model += dmd.mb.model;

            if (nd > 1) {
                LOG_TRC("%s:   - %s: %6" PRId64 " total, %6" PRId64 " used, %6" PRId64 " free vs. target of %6" PRId64 "\n",
                    __func__, dev_names[id].c_str(), dmd.total/MiB, projected_used/MiB, projected_free/MiB, margins[id]/MiB);
            }
        }
        assert(sum_free >= 0 && sum_projected_used >= 0);
        LOG_TRC("%s: projected to use %" PRId64 " MiB of device memory vs. %" PRId64 " MiB of free device memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (nd == 1) {
            if (projected_free_per_device[0] >= margins[0]) {
                LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of free device memory, no changes needed\n",
                    __func__, projected_free_per_device[0]/MiB, margins[0]/MiB);
                return;
            }
        } else {
            bool changes_needed = false;
            for (size_t id = 0; id < nd; id++) {
                if (projected_free_per_device[id] < margins[id]) {
                    changes_needed = true;
                    break;
                }
            }
            if (!changes_needed) {
                LOG_TRC("%s: targets for free memory can be met on all devices, no changes needed\n", __func__);
                return;
            }
        }
    }

    // step 2: try reducing memory use by reducing the context size

    {
        int64_t global_surplus = sum_projected_free;
        if (nd == 0) {
            global_surplus -= margins[0];
        } else {
            for (size_t id = 0; id < nd; id++) {
                global_surplus -= margins[id];
            }
        }
        if (global_surplus < 0) {
            if (nd <= 1) {
                LOG_TRC("%s: cannot meet free memory target of %" PRId64 " MiB, need to reduce device memory by %" PRId64 " MiB\n",
                    __func__, margins[0]/MiB, -global_surplus/MiB);
            } else {
                LOG_TRC(
                    "%s: cannot meet free memory targets on all devices, need to use %" PRId64 " MiB less in total\n",
                    __func__, -global_surplus/MiB);
            }
            if (n_ctx_auto) {
                if (n_ctx_max > n_ctx_min_total) {
                    int64_t sum_used_target = sum_free;
                    if (nd == 0) {
                        sum_used_target -= margins[0];
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_used_target -= margins[id];
                        }
                    }
                    if (nd > 1) {
                        // for multiple devices we need to be more conservative in terms of how much context we think can fit:
                        //   - for dense models only whole layers can be assigned to devices
                        //   - for MoE models only whole tensors can be assigned to devices, which we estimate to be <= 1/3 of a layer
                        //   - on average we expect a waste of 0.5 layers/tensors per device
                        //   - use slightly more than the expected average for nd devices to be safe
                        const int64_t model_per_layer = sum_projected_model / std::min(uint32_t(mparams->n_gpu_layers), hp_ngl);
                        sum_used_target -= (nd + 1) * model_per_layer / (hp_nex == 0 ? 2 : 6);
                    }

                    int64_t sum_projected_used_min_ctx = 0;
                    cparams->n_ctx = n_ctx_min_total;
                    dmds_t dmds_min_ctx = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                    add_extra_memory(dmds_min_ctx);
                    if (nd == 0) {
                        sum_projected_used_min_ctx = dmds_min_ctx.back().mb.total();
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_projected_used_min_ctx += dmds_min_ctx[id].mb.total();
                        }
                    }
                    if (sum_used_target > sum_projected_used_min_ctx) {
                        // linear interpolation between minimum and maximum context size:
                        cparams->n_ctx += (n_ctx_max - n_ctx_min_total) * (sum_used_target - sum_projected_used_min_ctx)
                            / (sum_projected_used - sum_projected_used_min_ctx);
                        // round down context for CUDA backend, keep it divisible by the number of streams:
                        const uint32_t align = 256 * n_streams;
                        cparams->n_ctx = std::max(cparams->n_ctx - cparams->n_ctx % align, n_ctx_min_total);

                        const int64_t bytes_per_ctx = (sum_projected_used - sum_projected_used_min_ctx) / (n_ctx_max - n_ctx_min_total);
                        const int64_t memory_reduction = (n_ctx_max - cparams->n_ctx) * bytes_per_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, n_ctx_max, cparams->n_ctx, memory_reduction/MiB);
                        if (nd <= 1) {
                            LOG_TRC("%s: entire model can be fit by reducing context\n", __func__);
                            return;
                        }
                        LOG_TRC("%s: entire model should be fit across devices by reducing context\n", __func__);
                    } else {
                        const int64_t memory_reduction = sum_projected_used - sum_projected_used_min_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, n_ctx_max, cparams->n_ctx, memory_reduction/MiB);
                    }
                } else {
                    if (n_ctx_min == UINT32_MAX) {
                        LOG_TRC("%s: user has requested full context size of %" PRIu32 " -> no change\n", __func__, n_ctx_max);
                    } else {
                        LOG_TRC("%s: default model context size is %" PRIu32 " which is <= the min. context size of %" PRIu32 " -> no change\n",
                            __func__, n_ctx_max, n_ctx_min_total);
                    }
                }
            } else {
                LOG_TRC("%s: context size set by user to %" PRIu32 " -> no change\n", __func__, cparams->n_ctx);
            }
        }
    }
    if (nd == 0) {
        throw common_params_fit_exception("was unable to fit model into system memory by reducing context, abort");
    }

    if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
        throw common_params_fit_exception("n_gpu_layers already set by user to " + std::to_string(mparams->n_gpu_layers) + ", abort");
    }
    if (nd > 1) {
        if (!tensor_split) {
            throw common_params_fit_exception("did not provide a buffer to write the tensor_split to, abort");
        }
        if (mparams->tensor_split) {
            for (size_t id = 0; id < nd; id++) {
                if (mparams->tensor_split[id] != 0.0f) {
                    throw common_params_fit_exception("model_params::tensor_split already set by user, abort");
                }
            }
        }
        if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
            throw common_params_fit_exception("changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented, abort");
        }
    }
    if (!tensor_buft_overrides) {
        throw common_params_fit_exception("did not provide buffer to set tensor_buft_overrides, abort");
    }
    if (mparams->tensor_buft_overrides && (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft)) {
        throw common_params_fit_exception("model_params::tensor_buft_overrides already set by user, abort");
    }

    // step 3: iteratively fill the back to front with "dense" layers
    //   - for a dense model simply fill full layers, giving each device a contiguous slice of the model
    //   - for a MoE model, same as dense model but with all MoE tensors in system memory

    // utility function that returns a static C string matching the tensors for a specific layer index and layer fraction:
    auto get_overflow_pattern = [&](const size_t il, const common_layer_fraction_t lf) -> const char * {
        constexpr size_t n_strings = 1000;
        if (il >= n_strings) {
            throw std::runtime_error("at most " + std::to_string(n_strings) + " model layers are supported");
        }
        switch (lf) {
            case LAYER_FRACTION_ATTN: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|up|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_UP: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_GATE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_down.*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_MOE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|down|gate_up|gate)_(ch|)exps";
                }
                return patterns[il].c_str();
            }
            default:
                GGML_ABORT("fatal error");
        }
    };

    struct ngl_t {
        uint32_t n_layer = 0; // number of total layers
        uint32_t n_part  = 0; // number of partial layers, <= n_layer

        // for the first partial layer varying parts can overflow, all further layers use LAYER_FRACTION_MOE:
        common_layer_fraction_t overflow_type = LAYER_FRACTION_MOE;

        uint32_t n_full() const {
            assert(n_layer >= n_part);
            return n_layer - n_part;
        }
    };

    const size_t ntbo = llama_max_tensor_buft_overrides();

    // utility function to set n_gpu_layers and tensor_split
    auto set_ngl_tensor_split_tbo = [&](
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts,
            llama_model_params & mparams) {
        mparams.n_gpu_layers = 0;
        for (size_t id = 0; id < nd; id++) {
            mparams.n_gpu_layers += ngl_per_device[id].n_layer;
            if (nd > 1) {
                tensor_split[id] = ngl_per_device[id].n_layer;
            }
        }
        assert(uint32_t(mparams.n_gpu_layers) <= hp_ngl + 1);
        uint32_t il0 = hp_ngl + 1 - mparams.n_gpu_layers; // start index for tensor buft overrides

        mparams.tensor_split = tensor_split;

        size_t itbo = 0;
        for (size_t id = 0; id < nd; id++) {
            il0 += ngl_per_device[id].n_full();
            for (uint32_t il = il0; il < il0 + ngl_per_device[id].n_part; il++) {
                if (itbo + 1 >= ntbo) {
                    tensor_buft_overrides[itbo].pattern = nullptr;
                    tensor_buft_overrides[itbo].buft    = nullptr;
                    itbo++;
                    mparams.tensor_buft_overrides = tensor_buft_overrides;
                    throw common_params_fit_exception("llama_max_tensor_buft_overrides() == "
                        + std::to_string(ntbo) + " is insufficient for model");
                }
                tensor_buft_overrides[itbo].pattern = get_overflow_pattern(il, il == il0 ? ngl_per_device[id].overflow_type : LAYER_FRACTION_MOE);
                tensor_buft_overrides[itbo].buft = il == il0 ? overflow_bufts[id] : ggml_backend_cpu_buffer_type();
                itbo++;
            }
            il0 += ngl_per_device[id].n_part;
        }
        tensor_buft_overrides[itbo].pattern = nullptr;
        tensor_buft_overrides[itbo].buft    = nullptr;
        itbo++;
        mparams.tensor_buft_overrides = tensor_buft_overrides;
    };

    // utility function that returns the memory use per device for given numbers of layers per device
    auto get_memory_for_layers = [&](
            const char * func_name,
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts) -> std::vector<int64_t> {
        llama_model_params mparams_copy = *mparams;
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, mparams_copy);

        dmds_t dmd_nl = common_get_device_memory_data_impl(
            path_model, &mparams_copy, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        add_extra_memory(dmd_nl);

        LOG_TRC("%s: memory for test allocation by device:\n", func_name);
        for (size_t id = 0; id < nd; id++) {
            const ngl_t & n = ngl_per_device[id];
            LOG_TRC(
                "%s: id=%zu, n_layer=%2" PRIu32 ", n_part=%2" PRIu32 ", overflow_type=%d, mem=%6" PRId64 " MiB\n",
                func_name, id, n.n_layer, n.n_part, int(n.overflow_type), dmd_nl[id].mb.total()/MiB);
        }

        std::vector<int64_t> ret;
        ret.reserve(nd);
        for (size_t id = 0; id < nd; id++) {
            ret.push_back(dmd_nl[id].mb.total());
        }
        return ret;
    };

    int64_t global_surplus_cpu_moe = 0;
    if (hp_nex > 0) {
        const static std::string pattern_moe_all = "blk\\.\\d+\\.ffn_(up|down|gate_up|gate)_(ch|)exps"; // matches all MoE tensors
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        tensor_buft_overrides[0] = {pattern_moe_all.c_str(), cpu_buft};
        tensor_buft_overrides[1] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;

        LOG_TRC("%s: getting device memory data with all MoE tensors moved to system memory:\n", __func__);
        dmds_t dmds_cpu_moe = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        add_extra_memory(dmds_cpu_moe);

        for (size_t id = 0; id < nd; id++) {
            global_surplus_cpu_moe += dmds_cpu_moe[id].free;
            global_surplus_cpu_moe -= int64_t(dmds_cpu_moe[id].mb.total()) + margins[id];
        }

        if (global_surplus_cpu_moe > 0) {
            LOG_TRC("%s: with only dense weights in device memory there is a total surplus of %" PRId64 " MiB\n",
                __func__, global_surplus_cpu_moe/MiB);
        } else {
            LOG_TRC("%s: with only dense weights in device memory there is still a total deficit of %" PRId64 " MiB\n",
                __func__, -global_surplus_cpu_moe/MiB);
        }

        // reset
        tensor_buft_overrides[0] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;
    }

    std::vector<int64_t> targets; // maximum acceptable memory use per device
    targets.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        targets.push_back(dmds_full[id].free - margins[id]);
        LOG_TRC("%s: id=%zu, target=%" PRId64 " MiB\n", __func__, id, targets[id]/MiB);
    }

    std::vector<ggml_backend_buffer_type_t> overflow_bufts; // which bufts the first partial layer of a device overflows to:
    overflow_bufts.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        overflow_bufts.push_back(ggml_backend_cpu_buffer_type());
    }

    std::vector<ngl_t> ngl_per_device(nd);
    std::vector<int64_t> mem = get_memory_for_layers(__func__, ngl_per_device, overflow_bufts);

    // optimize the number of layers per device using the method of false position:
    //   - ngl_per_device has 0 layers for each device, lower bound
    //   - try a "high" configuration where a device is given all unassigned layers
    //   - interpolate the memory use / layer between low and high linearly to get a guess where it meets our target
    //   - check memory use of our guess, replace either the low or high bound
    //   - once we only have a difference of a single layer, stop and return the lower bound that just barely still fits
    //   - the last device has the output layer, which cannot be a partial layer
    if (hp_nex == 0) {
        LOG_TRC("%s: filling dense layers back-to-front:\n", __func__);
    } else {
        LOG_TRC("%s: filling dense-only layers back-to-front:\n", __func__);
    }
    for (int id = nd - 1; id >= 0; id--) {
        uint32_t n_unassigned = hp_ngl + 1;
        for (size_t jd = id + 1; jd < nd; ++jd) {
            assert(n_unassigned >= ngl_per_device[jd].n_layer);
            n_unassigned -= ngl_per_device[jd].n_layer;
        }

        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        ngl_per_device_high[id].n_layer = n_unassigned;
        if (hp_nex > 0) {
            ngl_per_device_high[id].n_part = size_t(id) < nd - 1 ? ngl_per_device_high[id].n_layer : ngl_per_device_high[id].n_layer - 1;
        }
        if (ngl_per_device_high[id].n_layer > 0) {
            std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);
            if (mem_high[id] > targets[id]) {
                assert(ngl_per_device_high[id].n_layer > ngl_per_device[id].n_layer);
                uint32_t delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                LOG_TRC("%s: start filling device %" PRIu32 ", delta=%" PRIu32 "\n", __func__, id, delta);
                while (delta > 1) {
                    uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                    step_size = std::max(step_size, uint32_t(1));
                    step_size = std::min(step_size, delta - 1);

                    std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                    ngl_per_device_test[id].n_layer += step_size;
                    if (hp_nex) {
                        ngl_per_device_test[id].n_part += size_t(id) == nd - 1 && ngl_per_device_test[id].n_part == 0 ?
                            step_size - 1 : step_size; // the first layer is the output layer which must always be full
                    }
                    const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                    if (mem_test[id] <= targets[id]) {
                        ngl_per_device = ngl_per_device_test;
                        mem            = mem_test;
                        LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
                    } else {
                        ngl_per_device_high = ngl_per_device_test;
                        mem_high            = mem_test;
                        LOG_TRC("%s: set ngl_per_device_high[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device_high[id].n_layer);
                    }
                    delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                }
            } else {
                assert(ngl_per_device_high[id].n_layer == n_unassigned);
                ngl_per_device = ngl_per_device_high;
                mem            = mem_high;
                LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers, %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, mem[id]/MiB, projected_margin/MiB);
    }
    if (hp_nex == 0 || global_surplus_cpu_moe <= 0) {
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
        return;
    }

    // step 4: for a MoE model where all dense tensors fit,
    //     convert the dense-only layers in the back to full layers in the front until all devices are full
    // essentially the same procedure as for the dense-only layers except front-to-back
    // also, try fitting at least part of one more layer to reduce waste for "small" GPUs with e.g. 24 GiB VRAM

    size_t id_dense_start = nd;
    for (int id = nd - 1; id >= 0; id--) {
        if (ngl_per_device[id].n_layer > 0) {
            id_dense_start = id;
            continue;
        }
        break;
    }
    assert(id_dense_start < nd);

    LOG_TRC("%s: converting dense-only layers to full layers and filling them front-to-back with overflow to next device/system memory:\n", __func__);
    for (size_t id = 0; id <= id_dense_start && id_dense_start < nd; id++) {
        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        for (size_t jd = id_dense_start; jd < nd; jd++) {
            const uint32_t n_layer_move = jd < nd - 1 ? ngl_per_device_high[jd].n_layer : ngl_per_device_high[jd].n_layer - 1;
            ngl_per_device_high[id].n_layer += n_layer_move;
            ngl_per_device_high[jd].n_layer -= n_layer_move;
            ngl_per_device_high[jd].n_part = 0;
        }
        size_t id_dense_start_high = nd - 1;
        std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);

        if (mem_high[id] > targets[id]) {
            assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
            uint32_t delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            while (delta > 1) {
                uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                step_size = std::max(step_size, uint32_t(1));
                step_size = std::min(step_size, delta - 1);

                std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                size_t id_dense_start_test = id_dense_start;
                uint32_t n_converted_test = 0;
                for (;id_dense_start_test < nd; id_dense_start_test++) {
                    const uint32_t n_convert_jd = std::min(step_size - n_converted_test, ngl_per_device_test[id_dense_start_test].n_part);
                    ngl_per_device_test[id_dense_start_test].n_layer -= n_convert_jd;
                    ngl_per_device_test[id_dense_start_test].n_part -= n_convert_jd;
                    ngl_per_device_test[id].n_layer += n_convert_jd;
                    n_converted_test += n_convert_jd;

                    if (ngl_per_device_test[id_dense_start_test].n_part > 0) {
                        break;
                    }
                }
                const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                if (mem_test[id] <= targets[id]) {
                    ngl_per_device = ngl_per_device_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                } else {
                    ngl_per_device_high = ngl_per_device_test;
                    mem_high            = mem_test;
                    id_dense_start_high = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device_high[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start_high=%zu\n",
                        __func__, id, ngl_per_device_high[id].n_layer, ngl_per_device_high[id].n_part, id_dense_start_high);
                }
                assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
                delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            }
        } else {
            ngl_per_device = ngl_per_device_high;
            mem            = mem_high;
            id_dense_start = id_dense_start_high;
            LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
        }

        // try to fit at least part of one more layer
        if (ngl_per_device[id_dense_start].n_layer > (id < nd - 1 ? 0 : 1)) {
            std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
            size_t id_dense_start_test = id_dense_start;
            ngl_per_device_test[id_dense_start_test].n_layer--;
            ngl_per_device_test[id_dense_start_test].n_part--;
            ngl_per_device_test[id].n_layer++;
            ngl_per_device_test[id].n_part++;
            if (ngl_per_device_test[id_dense_start_test].n_part == 0) {
                id_dense_start_test++;
            }
            ngl_per_device_test[id].overflow_type = LAYER_FRACTION_UP;
            std::vector<ggml_backend_buffer_type_t> overflow_bufts_test = overflow_bufts;
            if (id < nd - 1) {
                overflow_bufts_test[id] = ggml_backend_dev_buffer_type(devs[id + 1]);
            }
            LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_UP\n", __func__);
            std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
            if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                ngl_per_device = ngl_per_device_test;
                overflow_bufts = overflow_bufts_test;
                mem            = mem_test;
                id_dense_start = id_dense_start_test;
                LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", UP), id_dense_start=%zu\n",
                    __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);

                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_GATE;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_GATE\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", GATE), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            } else {
                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_ATTN;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_ATTN\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", ATTN), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    // print info for devices that were not changed during the conversion from dense only to full layers:
    for (size_t id = id_dense_start + 1; id < nd; id++) {
        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
}

enum common_params_fit_status common_fit_params(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        float * tensor_split,
        llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins,
        uint32_t n_ctx_min,
        const common_fit_extra_model * extra,
        ggml_log_level log_level) {
    const int64_t t0_us = llama_time_us();
    common_params_fit_status status = COMMON_PARAMS_FIT_STATUS_SUCCESS;
    try {
        common_params_fit_impl(path_model, mparams, cparams, tensor_split, tensor_buft_overrides, margins, n_ctx_min, extra, log_level);
        LOG_TRC("%s: successfully fit params to free device memory\n", __func__);
    } catch (const common_params_fit_exception & e) {
        LOG_WRN("%s: failed to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_FAILURE;
    } catch (const std::runtime_error & e) {
        LOG_ERR("%s: encountered an error while trying to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_ERROR;
    }
    const int64_t t1_us = llama_time_us();
    LOG_TRC("%s: fitting params to free memory took %.2f seconds\n", __func__, (t1_us - t0_us) * 1e-6);
    return status;
}

void common_memory_breakdown_print(const struct llama_context * ctx) {
    //const auto & devices = ctx->get_model().devices;
    const auto * model = llama_get_model(ctx);

    std::vector<ggml_backend_dev_t> devices;
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devices.push_back(llama_model_get_device(model, i));
    }

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    std::vector<std::array<std::string, 9>> table_data;
    table_data.reserve(devices.size());

    // same data as the table below, for --log-jsonl consumers
    common_json rows = common_json::array();
    const std::string template_header = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
    const std::string template_gpu    = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
    const std::string template_other  = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

    table_data.push_back({template_header, "memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute", "unaccounted"});

    constexpr size_t MiB = 1024 * 1024;
    const std::vector<std::string> desc_prefixes_strip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

    // track seen buffer types to avoid double counting:
    std::set<ggml_backend_buffer_type_t> seen_buffer_types;

    // accumulative memory breakdown for each device and for host:
    std::vector<llama_memory_breakdown_data> mb_dev(devices.size());
    llama_memory_breakdown_data              mb_host;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (ggml_backend_buft_is_host(buft)) {
            mb_host.model   += mb.model;
            mb_host.context += mb.context;
            mb_host.compute += mb.compute;
            seen_buffer_types.insert(buft);
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            int i_dev = -1;
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i] == dev) {
                    i_dev = i;
                    break;
                }
            }
            if (i_dev != -1) {
                mb_dev[i_dev].model   += mb.model;
                mb_dev[i_dev].context += mb.context;
                mb_dev[i_dev].compute += mb.compute;
                seen_buffer_types.insert(buft);
                continue;
            }
        }
    }

    // print memory breakdown for each device:
    for (size_t i = 0; i < devices.size(); i++) {
        ggml_backend_dev_t dev = devices[i];
        llama_memory_breakdown_data mb = mb_dev[i];

        const std::string name = ggml_backend_dev_name(dev);
        std::string desc = ggml_backend_dev_description(dev);
        for (const std::string & prefix : desc_prefixes_strip) {
            if (desc.length() >= prefix.length() && desc.substr(0, prefix.length()) == prefix) {
                desc = desc.substr(prefix.length());
            }
        }

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        const size_t self = mb.model + mb.context + mb.compute;
        const int64_t unaccounted = static_cast<int64_t>(total) - static_cast<int64_t>(free) - static_cast<int64_t>(self);

        table_data.push_back({
            template_gpu,
            "  - " + name + " (" + desc + ")",
            std::to_string(total / MiB),
            std::to_string(free / MiB),
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            std::to_string(unaccounted / static_cast<int64_t>(MiB))});

        rows.push_back({
            {"kind",        "device"},
            {"name",        name},
            {"description", desc},
            {"total",       total / MiB},
            {"free",        free / MiB},
            {"self",        self / MiB},
            {"model",       mb.model / MiB},
            {"context",     mb.context / MiB},
            {"compute",     mb.compute / MiB},
            {"unaccounted", unaccounted / static_cast<int64_t>(MiB)},
        });
    }

    // print memory breakdown for host:
    {
        const size_t self = mb_host.model + mb_host.context + mb_host.compute;
        table_data.push_back({
            template_other,
            "  - Host",
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb_host.model / MiB),
            std::to_string(mb_host.context / MiB),
            std::to_string(mb_host.compute / MiB),
            ""}); // unaccounted

        rows.push_back({
            {"kind",    "host"},
            {"name",    "Host"},
            {"self",    self / MiB},
            {"model",   mb_host.model / MiB},
            {"context", mb_host.context / MiB},
            {"compute", mb_host.compute / MiB},
        });
    }

    // print memory breakdown for all remaining buffer types:
    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (seen_buffer_types.count(buft) == 1) {
            continue;
        }
        const std::string name = ggml_backend_buft_name(buft);
        const size_t self = mb.model + mb.context + mb.compute;
        table_data.push_back({
            template_other,
            "  - " + name,
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            ""}); // unaccounted

        rows.push_back({
            {"kind",    "buffer_type"},
            {"name",    name},
            {"self",    self / MiB},
            {"model",   mb.model / MiB},
            {"context", mb.context / MiB},
            {"compute", mb.compute / MiB},
        });

        seen_buffer_types.insert(buft);
    }

    for (size_t j = 1; j < table_data[0].size(); j++) {
        size_t max_len = 0;
        for (const auto & td : table_data) {
            max_len = std::max(max_len, td[j].length());
        }
        for (auto & td : table_data) {
            td[j].insert(j == 1 ? td[j].length() : 0, max_len - td[j].length(), ' ');
        }
    }
    for (const auto & td : table_data) {
        LOG_TRC(td[0].c_str(),
            __func__, td[1].c_str(), td[2].c_str(), td[3].c_str(), td[4].c_str(), td[5].c_str(),
            td[6].c_str(), td[7].c_str(), td[8].c_str());
    }

    LOG_JSON("fit_memory_breakdown", common_json({
        {"unit", "MiB"},
        {"rows", rows},
    }));
}

void common_fit_print(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams) {
    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    auto dmd = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);
    GGML_ASSERT(dmd.size() == devs.size() + 1);

    for (size_t id = 0; id < devs.size(); id++) {
        printf("%s ",  ggml_backend_dev_name(devs[id]));
        printf("%zu ", dmd[id].mb.model/1024/1024);
        printf("%zu ", dmd[id].mb.context/1024/1024);
        printf("%zu ", dmd[id].mb.compute/1024/1024);
        printf("\n");
    }

    printf("Host ");
    printf("%zu ", dmd.back().mb.model/1024/1024);
    printf("%zu ", dmd.back().mb.context/1024/1024);
    printf("%zu ", dmd.back().mb.compute/1024/1024);
    printf("\n");
}
