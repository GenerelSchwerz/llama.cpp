#pragma once

#include "ggml-cpp.h"
#include "llama.h"

#include <map>
#include <vector>

struct llama_model;
struct llama_vocab;
struct llm_graph_params;

class llama_draft_vocab {
public:
    llama_draft_vocab(const llama_model & model, const char * path);
    void apply(const llama_model & model, const llm_graph_params & params);

private:
    struct head_state {
        ggml_context_ptr context;
        ggml_backend_buffer_ptr buffer;
        ggml_tensor * weight = nullptr;
        ggml_tensor * ids = nullptr;
        bool warned = false;
    };

    std::vector<int32_t> selected;
    std::map<const ggml_tensor *, head_state> heads;
    bool boundary_warned = false;

    head_state & prepare(const ggml_tensor * head);
};

bool llama_draft_vocab_write(const llama_vocab & vocab, const int32_t * ids, size_t count, const char * path);
