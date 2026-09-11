#pragma once

#include "ggml-staged-input.h"
#include "llama.h"

#include <future>
#include <memory>
#include <vector>

struct llama_model;
struct llama_ubatch;
struct llm_graph_result;
class llama_memory_hybrid_idx_context;

class llama_staged_inputs {
public:
    static std::unique_ptr<llama_staged_inputs> create(const llama_model & model, ggml_backend_t backend, bool prefetch);
    ~llama_staged_inputs();
    void set_source(const llama_token * token, ggml_backend_event_t ready);
    void prepare(const llama_ubatch & ubatch, const llama_memory_hybrid_idx_context * memory);
    void finish();
    ggml_tensor * build_embedding(ggml_context * ctx, llm_graph_result * result);
    ggml_tensor * build_ple(ggml_context * ctx, llm_graph_result * result, const llama_memory_hybrid_idx_context * memory);

private:
    llama_staged_inputs(const llama_model & model, bool prefetch) : model(model), prefetch(prefetch) {}
    const llama_model & model;
    const bool prefetch;
    const ggml_staged_input_api * api = nullptr;
    void * embedding = nullptr;
    void * ple = nullptr;
    const llama_token * token = nullptr;
    ggml_backend_event_t ready = nullptr;
    std::future<void> pending;
};
