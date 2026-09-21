#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

enum llama_moe_placement_mode {
    LLAMA_MOE_PLACEMENT_ORDINARY_CPU,
    LLAMA_MOE_PLACEMENT_ORDINARY_DEVICE,
    LLAMA_MOE_PLACEMENT_RESIDUAL_CACHE,
    LLAMA_MOE_PLACEMENT_MIXED,
};

enum llama_moe_placement_reason {
    LLAMA_MOE_PLACEMENT_DEFAULT,
    LLAMA_MOE_PLACEMENT_USER_OVERRIDE,
    LLAMA_MOE_PLACEMENT_CACHE_LEGACY,
    LLAMA_MOE_PLACEMENT_CACHE_SELECTOR,
};

struct llama_moe_placement_bank {
    std::string name;
    uint32_t    role          = 0;
    uint32_t    status        = 0;
    int32_t     type          = 0;
    uint64_t    expert_stride = 0;
    uint64_t    tensor_bytes  = 0;
    llama_moe_placement_mode mode = LLAMA_MOE_PLACEMENT_ORDINARY_CPU;
    llama_moe_placement_reason reason = LLAMA_MOE_PLACEMENT_DEFAULT;
    int32_t owner_index = -1;
    std::string owner_id;
    std::string owner_name;
    std::string owner_canonical_id;
    std::string owner_identity_kind;
    std::string owner_backend;
    std::string actual_buft;
    bool        actual_buft_available = false;
    std::string resolved_class;
    int32_t winning_override_index = -1;
    std::string winning_override_pattern;
    std::string requested_buft;
    std::string selected_buft;
    std::string resolved_buft;
    size_t      allocation_estimate = 0;
    bool        allocation_estimate_available = false;
    std::string allocation_provenance;
};

struct llama_moe_placement_group {
    uint32_t                              semantic_index   = 0;
    int32_t                               layer            = -1;
    uint32_t                              layout           = 0;
    uint32_t                              domain           = 0;
    uint32_t                              n_experts        = 0;
    uint32_t                              top_k            = 0;
    uint32_t                              context_use_mask = 0;
    bool                                  route_present    = false;
    llama_moe_placement_mode              mode             = LLAMA_MOE_PLACEMENT_ORDINARY_CPU;
    int32_t                               owner_index      = -1;
    std::string                           owner_id;
    std::string                           owner_name;
    std::string                           owner_canonical_id;
    std::string                           placement_reason;
    int32_t                               cache_owner_index = -1;
    std::string                           cache_owner_canonical_id;
    size_t                                ordinary_allocation_estimate = 0;
    std::string                           ordinary_allocation_provenance;
    size_t                                cache_fixed_bytes    = 0;
    size_t                                cache_per_slot_bytes = 0;
    size_t                                cache_fixed_default_bytes = 0;
    size_t                                cache_per_slot_default_bytes = 0;
    size_t                                cache_fixed_mtp_bytes = 0;
    size_t                                cache_per_slot_mtp_bytes = 0;
    std::string                           cache_sizing_provenance;
    std::vector<llama_moe_placement_bank> banks;
};

struct llama_moe_placement_owner {
    uint32_t    selected_index = 0;
    std::string id;
    std::string name;
    std::string description;
    std::string backend;
    std::string canonical_id;
    std::string identity_kind;
    int32_t     slots                     = 0;
    size_t      cache_byte_cap            = 0;
    uint32_t    active_groups             = 0;
    uint32_t    active_context_mask       = 0;
    uint32_t    active_cache_groups       = 0;
    uint32_t    active_cache_context_mask = 0;
    size_t      cache_group_fixed_bytes   = 0;
    size_t      cache_group_per_slot_bytes = 0;
    size_t      cache_context_fixed_bytes = 0;
    size_t      cache_fixed_bytes         = 0;
    size_t      cache_per_slot_bytes      = 0;
    size_t      ordinary_allocation_estimate = 0;
    size_t      mandatory_host_staging_default_bytes = 0;
    size_t      mandatory_host_staging_mtp_bytes = 0;
    bool        mandatory_host_staging_available = false;
    std::string mandatory_host_staging_provenance;
};

struct llama_moe_shared_tensor {
    std::string name;
    size_t      tensor_bytes = 0;
    int32_t     type = 0;
    std::array<int64_t, GGML_MAX_DIMS> ne = {};
    std::array<size_t, GGML_MAX_DIMS>  nb = {};
    std::string resolved_buft;
    std::string resolved_class;
    std::string owner_canonical_id;
    std::string owner_identity_kind;
    std::string owner_backend;
    std::string storage_relation;
    bool        resolved_storage_available = false;
    bool        current_storage_available = false;
    std::string storage_provenance;
};

struct llama_moe_model_allocation {
    std::string resolved_class;
    std::string owner_canonical_id;
    std::string owner_identity_kind;
    std::string owner_backend;
    size_t      bytes = 0;
    bool        bytes_available = false;
    bool        current_allocation = false;
    std::string provenance;
};

struct llama_moe_placement_report {
    uint32_t                               schema_version = 1;
    std::string                            model_identity_kind;
    std::string                            model_identity;
    std::string                            model_identity_record;
    // Canonical identity of resolved model-side placement only. Context/KV,
    // target/draft pairing, runtime environment, and measurements belong in a
    // later full configuration identity and are intentionally not implied here.
    std::string                            placement_record;
    std::string                            placement_id;
    std::string                            placement_identity_kind;
    bool                                   uses_mmap = false;
    std::string                            direct_io_state = "none";
    bool                                   uses_mlock = false;
    bool                                   has_lazy_tensors = false;
    size_t                                 mandatory_host_staging_bytes = 0;
    std::vector<llama_moe_placement_group> groups;
    std::vector<llama_moe_placement_owner> owners;
    std::vector<llama_moe_shared_tensor>   shared_tensors;
    std::vector<llama_moe_model_allocation> model_allocations;
};

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Configure lazy row page advice (default: false) while idle, before the first staged decode attempt.
// Returns false without changes if the CPU extension is unavailable or staged inputs were already checked.
LLAMA_API bool llama_set_ple_prefetch(struct llama_context * ctx, bool enabled);

LLAMA_API bool llama_recurrent_sparse_snapshots_supported(const struct llama_context * ctx);
LLAMA_API bool llama_recurrent_set_sparse_snapshot_mode(
        struct llama_context * ctx, bool enabled, int32_t selected_token);

// Queue one decode using the previous backend-sampled token as device input.
// Requires synchronized output and supported backend sampling. On success, normal output access refers to the new decode.
// Stochastic sampling requires n_outputs_max_per_seq = 1. Discarding a queued position also requires restoring the sampler state saved before this call.
// Returns 0 if queued, 1 if unsupported without changing memory, or a negative value on decode failure.
LLAMA_API int32_t llama_decode_sampled(struct llama_context * ctx, llama_seq_id seq_id, llama_pos pos);

// Queue the next decode before waiting for the preceding sampled token, returned in previous.
// Uses the same return codes as llama_decode_sampled. On success, only previous is ready; the new decode can still be running.
LLAMA_API int32_t llama_decode_sampled_async(struct llama_context * ctx, llama_seq_id seq_id, llama_pos pos, llama_token * previous);

struct llama_sampled_decode_item {
    llama_seq_id seq_id;
    llama_pos pos;
};

// Queue one token per sequence from the preceding output at (seq_id, pos - 1).
// Items may select a subset of the preceding batch. previous follows item order.
// Return codes and completion rules match llama_decode_sampled_async.
LLAMA_API int32_t llama_decode_sampled_batch_async(
        struct llama_context * ctx, const llama_sampled_decode_item * items, int32_t n_items, llama_token * previous);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Returns an owned, read-only view of the model's resolved MoE placement.
// The result contains no tensor, buffer, graph, or backend pointers.
LLAMA_API llama_moe_placement_report llama_model_moe_placement(const struct llama_model * model);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);
// Returns 1 when shared, 0 for incompatible placement, and -1 with an empty borrower scheduler after failure.
LLAMA_API int32_t llama_attach_shared_workspace(
              struct llama_context * borrower,
              struct llama_context * owner);
LLAMA_API bool llama_contexts_share_workspace(
        const struct llama_context * ctx_a,
        const struct llama_context * ctx_b);

// Synchronize the context and ask capable backends to release cached transient physical mappings. This is a no-op unless live-context sizing is effective.
LLAMA_API uint64_t llama_trim_transient_memory(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
