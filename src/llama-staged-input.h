#pragma once

#include "ggml-staged-input.h"
#include "llama.h"

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct llama_model;
struct llama_ubatch;
struct llm_graph_result;
class llama_memory_hybrid_idx_context;

class llama_staged_inputs {
public:
    static std::unique_ptr<llama_staged_inputs> create(const llama_model & model, ggml_backend_t backend, bool prefetch, uint32_t n_tokens = 1, bool sampled = true);
    ~llama_staged_inputs();
    void set_source(const llama_token * token, ggml_backend_event_t ready);
    void prepare(const llama_ubatch & ubatch, const llama_memory_hybrid_idx_context * memory);
    void set_rows(uint32_t rows);
    void set_lookahead(const llama_ubatch * ubatch, const llama_memory_hybrid_idx_context * memory);
    void set_deferred(bool value) { deferred = value; }
    void submit();
    void finish();
    bool can_decode_sampled() const { return sampled; }
    uint32_t rows() const { return n_tokens; }
    ggml_tensor * build_embedding(ggml_context * ctx, llm_graph_result * result);
    ggml_tensor * build_ple(ggml_context * ctx, llm_graph_result * result, const llama_memory_hybrid_idx_context * memory);

private:
    void run();
    llama_staged_inputs(const llama_model & model, bool prefetch, uint32_t n_tokens, bool sampled) : model(model), prefetch(prefetch), capacity(n_tokens), n_tokens(n_tokens), sampled(sampled) {}
    const llama_model & model;
    const bool prefetch;
    const uint32_t capacity;
    uint32_t n_tokens;
    const bool sampled;
    const ggml_staged_input_api * api = nullptr;
    ggml_staged_input_consumed_t consumed = nullptr;
    void * embedding = nullptr;
    void * ple = nullptr;
    const llama_token * token = nullptr;
    ggml_backend_event_t ready = nullptr;
    std::future<void> pending;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable condition;
    std::packaged_task<void()> task;
    bool stopping = false;
    bool deferred = false, submit_supported = false, task_submitted = false;
    bool bounded_prefill = false;
    struct known_chunk {
        std::vector<llama_token> tokens, previous;
        std::vector<llama_pos> positions;
        std::vector<llama_seq_id> sequences;
        std::vector<float> embedding, ple;
    };
    known_chunk lookahead, collected;
    void collect_ple(const std::vector<llama_token> & tokens, const std::vector<llama_token> & previous, float * dst) const;
};
