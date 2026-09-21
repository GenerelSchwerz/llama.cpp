#pragma once

#include "ggml.h"
#include "llama.h"

#include <optional>
#include <string>
#include <vector>

struct llama_moe_placement_report;
struct common_params;

enum common_joint_role {
    COMMON_JOINT_ROLE_TARGET,
    COMMON_JOINT_ROLE_DRAFT,
    COMMON_JOINT_ROLE_MTP,
};

enum common_joint_sharing {
    COMMON_JOINT_SHARING_NONE,
    COMMON_JOINT_SHARING_BORROW_TARGET,
    COMMON_JOINT_SHARING_TARGET_MODEL,
};

enum common_joint_completeness {
    COMMON_JOINT_COMPLETENESS_COMPLETE,
    COMMON_JOINT_COMPLETENESS_INCOMPLETE,
    COMMON_JOINT_COMPLETENESS_ERROR,
};

enum common_joint_capacity {
    COMMON_JOINT_CAPACITY_WITHIN_LIMITS,
    COMMON_JOINT_CAPACITY_EXCEEDS_LIMITS,
    COMMON_JOINT_CAPACITY_UNKNOWN,
};

enum common_joint_bound {
    COMMON_JOINT_BOUND_EXACT_ESTIMATE,
    COMMON_JOINT_BOUND_UPPER_ESTIMATE,
    COMMON_JOINT_BOUND_UNKNOWN,
};

struct common_joint_device_limit {
    std::string canonical_id;
    size_t      capacity_bytes     = 0;
    size_t      margin_bytes       = 0;
    bool        capacity_available = true;
    std::string provenance         = "explicit_limit";
};

struct common_joint_component_request {
    std::string          path_model;
    // Required when cparams.samplers is non-empty. This identifies sampler
    // semantics without incorporating process-local sampler addresses.
    std::string          sampler_configuration_id;
    llama_model_params   mparams  = llama_model_default_params();
    llama_context_params cparams  = llama_context_default_params();
    common_joint_role    role     = COMMON_JOINT_ROLE_TARGET;
    common_joint_sharing sharing  = COMMON_JOINT_SHARING_NONE;
    bool                 required = true;
};

struct common_joint_measurement_request {
    common_joint_component_request              target;
    std::vector<common_joint_component_request> extras;
    std::vector<common_joint_device_limit>      device_limits;
    std::vector<size_t>                         target_device_margins;
    size_t                                      default_device_margin = 0;
    std::optional<size_t>                       host_capacity_bytes;
    size_t                                      host_margin_bytes        = 0;
    std::string                                 host_capacity_provenance = "explicit_host_limit";
    // Explicit opt-in: library callers otherwise receive unknown capacity.
    bool                                        observe_device_capacity  = false;
    ggml_log_level                              log_level                = GGML_LOG_LEVEL_ERROR;
    const common_params *                       runtime_params           = nullptr;
};

struct common_joint_memory {
    size_t model             = 0;
    size_t context           = 0;
    size_t compute           = 0;
    size_t staging           = 0;
    bool   staging_available = true;
};

struct common_joint_component_owner {
    std::string         canonical_id;
    std::string         identity_kind;
    std::string         backend;
    std::string         name;
    bool                host = false;
    common_joint_memory memory;
    size_t              lower_bound_excluded_bytes = 0;
    std::string         provenance;
};

struct common_joint_sharing_record {
    std::string tensor_name;
    std::string owner_canonical_id;
    std::string owner_identity_kind;
    size_t      tensor_payload_bytes = 0;
    std::string relation;
    std::string provenance;
};

struct common_joint_component_measurement {
    common_joint_role                         role         = COMMON_JOINT_ROLE_TARGET;
    common_joint_sharing                      sharing      = COMMON_JOINT_SHARING_NONE;
    bool                                      required     = true;
    common_joint_completeness                 completeness = COMMON_JOINT_COMPLETENESS_ERROR;
    std::string                               configuration_id;
    std::string                               configuration_record;
    std::string                               model_identity;
    std::string                               model_identity_kind;
    std::string                               placement_id;
    std::string                               placement_identity_kind;
    size_t                                    shared_model_bytes_deduplicated = 0;
    size_t                                    shared_tensor_payload_bytes     = 0;
    std::string                               sharing_provenance;
    common_joint_bound                        bound = COMMON_JOINT_BOUND_UNKNOWN;
    std::string                               memory_provenance;
    std::vector<common_joint_component_owner> owners;
    std::vector<common_joint_sharing_record>  sharing_records;
    std::vector<std::string>                  diagnostics;
};

struct common_joint_owner_measurement {
    std::string         canonical_id;
    std::string         identity_kind;
    std::string         backend;
    std::string         name;
    bool                host = false;
    common_joint_memory memory;
    size_t              lower_bound_excluded_bytes = 0;
    size_t              required_lower_bytes       = 0;
    size_t              required_upper_bytes       = 0;
    common_joint_bound  bound                      = COMMON_JOINT_BOUND_UNKNOWN;
    std::string         memory_provenance;
    bool                capacity_available = false;
    size_t              capacity_bytes     = 0;
    size_t              total_bytes        = 0;
    size_t              margin_bytes       = 0;
    int64_t             slack_bytes        = 0;
    std::string         capacity_provenance;
};

struct common_joint_measurement {
    uint32_t                                        schema_version      = 1;
    common_joint_completeness                       completeness        = COMMON_JOINT_COMPLETENESS_ERROR;
    common_joint_capacity                           capacity            = COMMON_JOINT_CAPACITY_UNKNOWN;
    bool                                            admission_qualified = false;
    std::string                                     configuration_id;
    std::string                                     configuration_record;
    std::vector<common_joint_component_measurement> components;
    std::vector<common_joint_owner_measurement>     owners;
    std::vector<std::string>                        diagnostics;
};

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occurred, e.g. because no model could be found at the specified path
};

// a second model that shares the devices of the main model, e.g. a draft model
//   - its context follows the context of the main model, so its memory is measured again whenever that context changes
//   - shares_model tells the fit that the weights are already counted in the main model, as for an MTP context
struct common_fit_extra_model {
    const char * path_model;
    llama_model_params * mparams;
    llama_context_params * cparams;
    bool shares_model;
};

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global llama logger state
//   - only parameters that have the same value as in llama_default_model_params are modified
//     with the exception of the context size which is modified if and only if equal to 0
common_params_fit_status common_fit_params(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams,
                              float * tensor_split,          // writable buffer for tensor split, needs at least llama_max_devices elements
   llama_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
                             size_t * margins,               // margins of memory to leave per device in bytes
                           uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
      const common_fit_extra_model * extra,                  // model to fit alongside the main one, nullptr if there is none
                     ggml_log_level   log_level);            // minimum log level to print during fitting, lower levels go to debug log

// print estimated memory to stdout
void common_fit_print(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams);

void common_memory_breakdown_print(const llama_context * ctx);

// Formats the owned model-side placement snapshot without observing or
// synchronizing live backend work. JSON is a single compact object.
std::string common_moe_placement_report_json(
        const llama_moe_placement_report & report,
        const llama_model_params & mparams,
        const llama_context_params & cparams,
        const common_params * runtime_params = nullptr);
std::string common_moe_placement_report_human(
        const llama_moe_placement_report & report,
        const llama_model_params & mparams,
        const llama_context_params & cparams,
        const common_params * runtime_params = nullptr);

// Measures a frozen target plus its active draft/MTP components without
// changing caller parameters, loading tensor payloads, or running inference.
// This function temporarily replaces the global llama logger and is therefore
// subject to the same thread-safety restriction as common_fit_params().
common_joint_measurement common_measure_joint_configuration(const common_joint_measurement_request & request);

// Builds the current target plus active model-backed speculative arrangement
// from common_params and measures it with all enabled extras required.
common_joint_measurement common_measure_joint_configuration(const common_params & params,
                                                            ggml_log_level        log_level = GGML_LOG_LEVEL_ERROR,
                                                            bool                  observe_device_capacity = false);

std::string common_joint_measurement_json(const common_joint_measurement & measurement);
std::string common_joint_measurement_human(const common_joint_measurement & measurement);

struct common_device_memory_data {
    int64_t total;
    int64_t free;
    size_t  model;
    size_t  context;
    size_t  compute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// Load a model + context with no_alloc and return the per-device memory breakdown.
common_device_memory_data_vec common_get_device_memory_data(
                         const char * path_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     ggml_log_level   log_level);
