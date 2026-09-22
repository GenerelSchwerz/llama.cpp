#include "llama-model.h"

#include "llama-arch.h"
#include "llama-ext.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-cparams.h"
#include "llama-model-loader.h"
#include "llama-version.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsa-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include "llama.h"
#include "models/models.h"
#include "hash/hash.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-backend-moe.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <functional>
#include <locale>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

bool llama_internal_model_no_alloc(const llama_model * model);

static bool is_moe_cache_source_buft(ggml_backend_buffer_type_t buft);

static llama_model * llama_model_mapping(llm_arch arch, const llama_model_params & params) {
    switch (arch) {
        case LLM_ARCH_CLIP:
            return new llama_model_clip(params);
        case LLM_ARCH_LLAMA:
            return new llama_model_llama(params);
        case LLM_ARCH_LLAMA4:
            return new llama_model_llama4(params);
        case LLM_ARCH_LLAMA_EMBED:
            return new llama_model_llama_embed(params);
        case LLM_ARCH_MAINCODER:
            return new llama_model_maincoder(params);
        case LLM_ARCH_TALKIE:
            return new llama_model_talkie(params);
        case LLM_ARCH_DECI:
            return new llama_model_deci(params);
        case LLM_ARCH_BAICHUAN:
            return new llama_model_baichuan(params);
        case LLM_ARCH_FALCON:
            return new llama_model_falcon(params);
        case LLM_ARCH_GROK:
            return new llama_model_grok(params);
        case LLM_ARCH_STARCODER:
            return new llama_model_starcoder(params);
        case LLM_ARCH_REFACT:
            return new llama_model_refact(params);
        case LLM_ARCH_BERT:
            return new llama_model_bert(params);
        case LLM_ARCH_JINA_BERT_V2:
            return new llama_model_jina_bert_v2(params);
        case LLM_ARCH_JINA_BERT_V3:
            return new llama_model_jina_bert_v3(params);
        case LLM_ARCH_NOMIC_BERT:
            return new llama_model_nomic_bert(params);
        case LLM_ARCH_NOMIC_BERT_MOE:
            return new llama_model_nomic_bert_moe(params);
        case LLM_ARCH_MODERN_BERT:
            return new llama_model_modern_bert(params);
        case LLM_ARCH_NEO_BERT:
            return new llama_model_neo_bert(params);
        case LLM_ARCH_EUROBERT:
            return new llama_model_eurobert(params);
        case LLM_ARCH_BLOOM:
            return new llama_model_bloom(params);
        case LLM_ARCH_MPT:
            return new llama_model_mpt(params);
        case LLM_ARCH_STABLELM:
            return new llama_model_stablelm(params);
        case LLM_ARCH_MELLUM:
            return new llama_model_mellum(params);
        case LLM_ARCH_NANBEIGE:
            return new llama_model_nanbeige(params);
        case LLM_ARCH_QWEN:
            return new llama_model_qwen(params);
        case LLM_ARCH_QWEN2:
            return new llama_model_qwen2(params);
        case LLM_ARCH_DREAM:
            return new llama_model_dream(params);
        case LLM_ARCH_LLADA:
            return new llama_model_llada(params);
        case LLM_ARCH_LLADA_MOE:
            return new llama_model_llada_moe(params);
        case LLM_ARCH_RND1:
            return new llama_model_rnd1(params);
        case LLM_ARCH_QWEN2VL:
            return new llama_model_qwen2vl(params);
        case LLM_ARCH_QWEN2MOE:
            return new llama_model_qwen2moe(params);
        case LLM_ARCH_QWEN3:
            return new llama_model_qwen3(params);
        case LLM_ARCH_QWEN3MOE:
            return new llama_model_qwen3moe(params);
        case LLM_ARCH_QWEN3VL:
            return new llama_model_qwen3vl(params);
        case LLM_ARCH_QWEN3VLMOE:
            return new llama_model_qwen3vlmoe(params);
        case LLM_ARCH_QWEN3TTS:
            return new llama_model_qwen3tts(params);
        case LLM_ARCH_POCKETTTS:
            return new llama_model_pockettts(params);
        case LLM_ARCH_PHI2:
            return new llama_model_phi2(params);
        case LLM_ARCH_PHI3:
            return new llama_model_phi3(params);
        case LLM_ARCH_PHIMOE:
            return new llama_model_phimoe(params);
        case LLM_ARCH_PLAMO:
            return new llama_model_plamo(params);
        case LLM_ARCH_PLAMO2:
            return new llama_model_plamo2(params);
        case LLM_ARCH_PLAMO3:
            return new llama_model_plamo3(params);
        case LLM_ARCH_GPT2:
            return new llama_model_gpt2(params);
        case LLM_ARCH_CODESHELL:
            return new llama_model_codeshell(params);
        case LLM_ARCH_ORION:
            return new llama_model_orion(params);
        case LLM_ARCH_INTERNLM2:
            return new llama_model_internlm2(params);
        case LLM_ARCH_MINICPM3:
            return new llama_model_minicpm3(params);
        case LLM_ARCH_GEMMA:
            return new llama_model_gemma(params);
        case LLM_ARCH_GEMMA2:
            return new llama_model_gemma2(params);
        case LLM_ARCH_GEMMA3:
            return new llama_model_gemma3(params);
        case LLM_ARCH_GEMMA3N:
            return new llama_model_gemma3n(params);
        case LLM_ARCH_GEMMA4:
            return new llama_model_gemma4(params);
        case LLM_ARCH_GEMMA4_ASSISTANT:
            return new llama_model_gemma4_assistant(params);
        case LLM_ARCH_GEMMA_EMBEDDING:
            return new llama_model_gemma_embedding(params);
        case LLM_ARCH_STARCODER2:
            return new llama_model_starcoder2(params);
        case LLM_ARCH_MAMBA:
            return new llama_model_mamba(params);
        case LLM_ARCH_MAMBA2:
            return new llama_model_mamba2(params);
        case LLM_ARCH_MAPLE:
            return new llama_model_maple(params);
        case LLM_ARCH_JAMBA:
            return new llama_model_jamba(params);
        case LLM_ARCH_XVERSE:
            return new llama_model_xverse(params);
        case LLM_ARCH_COMMAND_R:
            return new llama_model_command_r(params);
        case LLM_ARCH_COHERE2:
            return new llama_model_cohere2(params);
        case LLM_ARCH_COHERE2MOE:
            return new llama_model_cohere2moe(params);
        case LLM_ARCH_DBRX:
            return new llama_model_dbrx(params);
        case LLM_ARCH_OLMO:
            return new llama_model_olmo(params);
        case LLM_ARCH_OLMO2:
            return new llama_model_olmo2(params);
        case LLM_ARCH_OLMOE:
            return new llama_model_olmoe(params);
        case LLM_ARCH_MUSE_GLIMMER:
            return new llama_model_muse_glimmer(params);
        case LLM_ARCH_OPENELM:
            return new llama_model_openelm(params);
        case LLM_ARCH_GPTNEOX:
            return new llama_model_gptneox(params);
        case LLM_ARCH_ARCTIC:
            return new llama_model_arctic(params);
        case LLM_ARCH_DEEPSEEK:
            return new llama_model_deepseek(params);
        case LLM_ARCH_DEEPSEEK2:
            return new llama_model_deepseek2(params);
        case LLM_ARCH_DEEPSEEK2OCR:
            return new llama_model_deepseek2ocr(params);
        case LLM_ARCH_DEEPSEEK32:
            return new llama_model_deepseek32(params);
        case LLM_ARCH_DOTS3NOTE:
            return new llama_model_dots3note(params);
        case LLM_ARCH_DEEPSEEK4:
            return new llama_model_deepseek4(params);
        case LLM_ARCH_GLM_DSA:
            return new llama_model_glm_dsa(params);
        case LLM_ARCH_MISTRAL4:
            return new llama_model_mistral4(params);
        case LLM_ARCH_CHATGLM:
            return new llama_model_chatglm(params);
        case LLM_ARCH_GLM4:
            return new llama_model_glm4(params);
        case LLM_ARCH_GLM4_MOE:
            return new llama_model_glm4_moe(params);
        case LLM_ARCH_BITNET:
            return new llama_model_bitnet(params);
        case LLM_ARCH_T5:
            return new llama_model_t5(params);
        case LLM_ARCH_T5ENCODER:
            return new llama_model_t5encoder(params);
        case LLM_ARCH_JAIS:
            return new llama_model_jais(params);
        case LLM_ARCH_JAIS2:
            return new llama_model_jais2(params);
        case LLM_ARCH_NEMOTRON:
            return new llama_model_nemotron(params);
        case LLM_ARCH_NEMOTRON_H:
            return new llama_model_nemotron_h(params);
        case LLM_ARCH_NEMOTRON_H_MOE:
            return new llama_model_nemotron_h_moe(params);
        case LLM_ARCH_EXAONE:
            return new llama_model_exaone(params);
        case LLM_ARCH_EXAONE4:
            return new llama_model_exaone4(params);
        case LLM_ARCH_EXAONE_MOE:
            return new llama_model_exaone_moe(params);
        case LLM_ARCH_RWKV6:
            return new llama_model_rwkv6(params);
        case LLM_ARCH_RWKV6QWEN2:
            return new llama_model_rwkv6qwen2(params);
        case LLM_ARCH_RWKV7:
            return new llama_model_rwkv7(params);
        case LLM_ARCH_ARWKV7:
            return new llama_model_arwkv7(params);
        case LLM_ARCH_GRANITE:
            return new llama_model_granite(params);
        case LLM_ARCH_GRANITE_MOE:
            return new llama_model_granite_moe(params);
        case LLM_ARCH_GRANITE_SWITCH:
            return new llama_model_granite_switch(params);
        case LLM_ARCH_MINICPM:
            return new llama_model_minicpm(params);
        case LLM_ARCH_GRANITE_HYBRID:
            return new llama_model_granite_hybrid(params);
        case LLM_ARCH_GRANITE_SWA:
            return new llama_model_granite_swa(params);
        case LLM_ARCH_CHAMELEON:
            return new llama_model_chameleon(params);
        case LLM_ARCH_WAVTOKENIZER_DEC:
            return new llama_model_wavtokenizer_dec(params);
        case LLM_ARCH_PLM:
            return new llama_model_plm(params);
        case LLM_ARCH_BAILINGMOE:
            return new llama_model_bailingmoe(params);
        case LLM_ARCH_BAILINGMOE2:
            return new llama_model_bailingmoe2(params);
        case LLM_ARCH_BAILINGMOE3:
            return new llama_model_bailingmoe3(params);
        case LLM_ARCH_SEED_OSS:
            return new llama_model_seed_oss(params);
        case LLM_ARCH_DOTS1:
            return new llama_model_dots1(params);
        case LLM_ARCH_ARCEE:
            return new llama_model_arcee(params);
        case LLM_ARCH_AFMOE:
            return new llama_model_afmoe(params);
        case LLM_ARCH_LAGUNA:
            return new llama_model_laguna(params);
        case LLM_ARCH_ERNIE4_5:
            return new llama_model_ernie4_5(params);
        case LLM_ARCH_ERNIE4_5_MOE:
            return new llama_model_ernie4_5_moe(params);
        case LLM_ARCH_PADDLEOCR:
            return new llama_model_paddleocr(params);
        case LLM_ARCH_HUNYUAN_MOE:
            return new llama_model_hunyuan_moe(params);
        case LLM_ARCH_HUNYUAN_VL:
            return new llama_model_hunyuan_vl(params);
        case LLM_ARCH_HUNYUAN_DENSE:
            return new llama_model_hunyuan_dense(params);
        case LLM_ARCH_HY_V3:
            return new llama_model_hy_v3(params);
        case LLM_ARCH_HY_V4:
            return new llama_model_hy_v4(params);
        case LLM_ARCH_SMOLLM3:
            return new llama_model_smollm3(params);
        case LLM_ARCH_OPENAI_MOE:
            return new llama_model_openai_moe(params);
        case LLM_ARCH_FALCON_H1:
            return new llama_model_falcon_h1(params);
        case LLM_ARCH_LFM2:
            return new llama_model_lfm2(params);
        case LLM_ARCH_LFM2MOE:
            return new llama_model_lfm2moe(params);
        case LLM_ARCH_SMALLTHINKER:
            return new llama_model_smallthinker(params);
        case LLM_ARCH_GROVEMOE:
            return new llama_model_grovemoe(params);
        case LLM_ARCH_APERTUS:
            return new llama_model_apertus(params);
        case LLM_ARCH_MINIMAX_01:
            return new llama_model_minimax_01(params);
        case LLM_ARCH_MINIMAX_M2:
            return new llama_model_minimax_m2(params);
        case LLM_ARCH_MINIMAX_M3:
            return new llama_model_minimax_m3(params);
        case LLM_ARCH_HRM_TEXT:
            return new llama_model_hrm_text(params);
        case LLM_ARCH_COGVLM:
            return new llama_model_cogvlm(params);
        case LLM_ARCH_PANGU_EMBED:
            return new llama_model_pangu_embed(params);
        case LLM_ARCH_QWEN3NEXT:
            return new llama_model_qwen3next(params);
        case LLM_ARCH_QWEN35:
            return new llama_model_qwen35(params);
        case LLM_ARCH_QWEN35MOE:
            return new llama_model_qwen35moe(params);
        case LLM_ARCH_QWEN4EXP:
            return new llama_model_qwen4exp(params);
        case LLM_ARCH_MISTRAL3:
            return new llama_model_mistral3(params);
        case LLM_ARCH_EAGLE3:
            return new llama_model_eagle3(params);
        case LLM_ARCH_DFLASH:
            return new llama_model_dflash(params);
        case LLM_ARCH_MIMO2:
            return new llama_model_mimo2(params);
        case LLM_ARCH_KIMI_LINEAR:
            return new llama_model_kimi_linear(params);
        case LLM_ARCH_KIMI_K3:
            return new llama_model_kimi_k3(params);
        case LLM_ARCH_STEP35:
            return new llama_model_step35(params);
        case LLM_ARCH_SPARK2_5:
            return new llama_model_spark2_5(params);
        default:
            throw std::runtime_error(std::string("unsupported model architecture: '") + llm_arch_name(arch) + "'");
    }

}

llama_model * llama_model_create(llm_arch arch, const llama_model_params & params) {
    if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR &&
            (params.moe_expert_cache_slots > 0 || params.n_moe_expert_cache_byte_budgets != 0)) {
        throw std::runtime_error("MoE expert caching does not support tensor split; use --split-mode layer or --moe-expert-cache-size 0");
    }
    llama_model * model = llama_model_mapping(arch, params);

    if (model != nullptr) {
        model->arch = arch;
        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR && !llm_arch_supports_sm_tensor(arch)) {
            throw std::runtime_error(std::string("LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '") + llm_arch_name(arch) + "'");
        }
    }

    return model;
}

llama_model * llama_model_create(llama_model_loader & ml, const llama_model_params & params) {
    llm_arch arch = ml.get_arch();
    if (arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }

    return llama_model_create(arch, params);
}

struct ggml_backend_meta_split_state llama_meta_device_get_split_state(const struct ggml_tensor * tensor, void * userdata) {
    const llama_meta_device_get_split_state_userdata * ud = (const llama_meta_device_get_split_state_userdata *) userdata;
    const llama_hparams & hparams = ud->model->hparams;
    const std::string tensor_name = tensor->name;
    const bool is_dsv4 = ud->model->arch == LLM_ARCH_DEEPSEEK4 ||
        (ud->model->arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0);

    static const std::regex pattern_q_weight        ("blk\\.\\d*\\.attn_q.weight");
    static const std::regex pattern_kv_weight       ("blk\\.\\d*\\.attn_(k|v).weight");
    static const std::regex pattern_qkv_weight      ("blk\\.\\d*\\.attn_qkv.weight");
    static const std::regex pattern_q_bias          ("blk\\.\\d*\\.attn_q\\.bias");
    static const std::regex pattern_kv_bias         ("blk\\.\\d*\\.attn_(k|v)\\.bias");
    static const std::regex pattern_qkv_bias        ("blk\\.\\d*\\.attn_qkv.bias");
    static const std::regex pattern_qk_norm         ("blk\\.\\d*\\.attn_(q|k)_norm\\.weight");
    static const std::regex pattern_kv_cache        ("cache_(k|v)_l\\d*");
    static const std::regex pattern_idx_cache       ("cache_idx_(k|v)_l\\d*");
    static const std::regex pattern_dsv4_state      ("dsv4_(csa|hca|lid)_state_(kv|score)_l\\d*");
    static const std::regex pattern_attn_sinks      ("blk\\.\\d*\\.attn_sinks.weight");
    static const std::regex pattern_attn_out_weight ("blk\\.\\d*\\.attn_output.weight");
    static const std::regex pattern_attn_out_bias   ("blk\\.\\d*\\.attn_output.bias");
    static const std::regex pattern_attn_out_a_weight("blk\\.\\d*\\.attn_output_a\\.weight");
    static const std::regex pattern_attn_out_b_weight("blk\\.\\d*\\.attn_output_b\\.weight");
    static const std::regex pattern_attn_q_b_weight ("blk\\.\\d*\\.attn_q_b\\.weight");
    static const std::regex pattern_attn_gate_weight("blk\\.\\d*\\.attn_gate.weight");

    static const std::regex pattern_ssm_dt          ("blk\\.\\d*\\.ssm_dt.bias");
    static const std::regex pattern_ssm_a           ("blk\\.\\d*\\.ssm_a");
    static const std::regex pattern_ssm_alpha       ("blk\\.\\d*\\.ssm_alpha.weight");
    static const std::regex pattern_ssm_beta        ("blk\\.\\d*\\.ssm_beta.weight");
    static const std::regex pattern_ssm_beta_alpha  ("blk\\.\\d*\\.ssm_ba.weight");
    static const std::regex pattern_r_cache         ("cache_r_l\\d*");
    static const std::regex pattern_ple_r_cache     ("cache_ple_r_l\\d*");
    static const std::regex pattern_s_cache         ("cache_s_l\\d*");
    static const std::regex pattern_ssm_conv1d      ("blk\\.\\d*\\.ssm_conv1d.weight");
    static const std::regex pattern_ssm_out_weight  ("blk\\.\\d*\\.ssm_out.weight");

    static const std::regex pattern_ffn_up_weight     ("blk\\.\\d*\\.ffn_up(_exps)?.weight");
    static const std::regex pattern_ffn_up_bias       ("blk\\.\\d*\\.ffn_up(_exps)?.bias");
    static const std::regex pattern_ffn_gate_weight   ("blk\\.\\d*\\.ffn_gate(_exps)?.weight");
    static const std::regex pattern_ffn_gate_bias     ("blk\\.\\d*\\.ffn_gate(_exps)?.bias");
    static const std::regex pattern_ffn_gate_up_weight("blk\\.\\d*\\.ffn_gate_up(_exps)?.weight");
    static const std::regex pattern_ffn_down_weight   ("blk\\.\\d*\\.ffn_down(_exps)?.weight");
    static const std::regex pattern_ffn_down_bias         ("blk\\.\\d*\\.ffn_down.bias");
    static const std::regex pattern_ffn_down_exps_bias    ("blk\\.\\d*\\.ffn_down_exps.bias");
    static const std::regex pattern_ffn_up_shexp_weight   ("blk\\.\\d*\\.ffn_up_shexp.weight");
    static const std::regex pattern_ffn_gate_shexp_weight ("blk\\.\\d*\\.ffn_gate_shexp.weight");
    static const std::regex pattern_ffn_down_shexp_weight ("blk\\.\\d*\\.ffn_down_shexp.weight");

    static const std::regex pattern_output_weight("output\\.weight");
    static const std::regex pattern_output_bias  ("output\\.bias");

    struct tensor_config {
        ggml_backend_meta_split_axis axis;

        const ggml_tensor * tensor_axis_0;

        uint32_t il;
        size_t   rotation; // when assigning tensor slices, rotate how the rounding is done for more even allocation
    };

    auto get_tensor_config_impl = [&](
                const ggml_backend_meta_split_axis axis, const std::string & suffix = "", const std::string & suffix_fallback = "") -> tensor_config {
        // the layers in a tensor can be inhomogeneous, if the pattern is cleanly divided by the number of GPUs there can be aliasing effects,
        //     count only the same type of previous layers to avoid this
        auto get_il_eff = [&](const size_t il){
            size_t ret = 0;
            const bool il_is_recr = hparams.is_recr(il);
            const bool il_is_swa  = hparams.is_swa(il);
            for (size_t il_prev = 0; il_prev < il; il_prev++) {
                ret += hparams.is_recr(il_prev) == il_is_recr && hparams.is_swa(il_prev) == il_is_swa;
            }
            return ret;
        };

        uint32_t il;
        std::string prefix;
        size_t rotation;
        if (tensor_name.substr(0, 4) == "blk.") {
            const size_t length_prefix = tensor_name.find('.', 4);
            GGML_ASSERT(length_prefix != std::string::npos);
            prefix = tensor_name.substr(0, length_prefix + 1);
            il = std::stoull(tensor_name.substr(4, length_prefix));
            rotation = get_il_eff(il) % ud->n_devices;
        } else if (tensor_name.substr(0, 6) == "cache_") {
            const size_t layer_index_start = tensor_name.find("_l", 6);
            GGML_ASSERT(layer_index_start != std::string::npos);
            il = std::stoull(tensor_name.substr(layer_index_start + 2));
            prefix = "blk." + std::to_string(il) + ".";
            rotation = get_il_eff(il) % ud->n_devices;
        } else {
            il = 0;
            rotation = hparams.n_layer() % ud->n_devices;
        }
        const ggml_tensor * tensor_axis_0 = suffix.empty() ? tensor : ud->model->get_tensor((prefix + suffix).c_str());
        if (tensor_axis_0 == nullptr) {
            GGML_ASSERT(!suffix_fallback.empty());
            tensor_axis_0 = ud->model->get_tensor((prefix + suffix_fallback).c_str());
        }
        GGML_ASSERT(tensor_axis_0 != nullptr);
        return {axis, tensor_axis_0, il, rotation};
    };

    auto get_tensor_config = [&]() -> tensor_config {
        if (ud->model->arch == LLM_ARCH_HRM_TEXT) {
            // aliased cache slots cannot satisfy the meta-split invariants, so replicate all tensors
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, tensor, 0, 0};
        }
        if (is_dsv4) {
            if (std::regex_match(tensor_name, pattern_kv_cache) ||
                    std::regex_match(tensor_name, pattern_dsv4_state)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            }
            if (std::regex_match(tensor_name, pattern_attn_sinks)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output_a.weight");
            }
            if (std::regex_match(tensor_name, pattern_attn_q_b_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output_a.weight");
            }
            if (std::regex_match(tensor_name, pattern_attn_out_a_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_2);
            }
            if (std::regex_match(tensor_name, pattern_attn_out_b_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
            }
            if (std::regex_match(tensor_name, pattern_ffn_up_shexp_weight) ||
                    std::regex_match(tensor_name, pattern_ffn_gate_shexp_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down_shexp.weight");
            }
            if (std::regex_match(tensor_name, pattern_ffn_down_shexp_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down_shexp.weight");
            }
        }

        // the qsa indexer has one key head and its projections are mirrored, so its cache cannot be split
        if (std::regex_match(tensor_name, pattern_idx_cache)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        // the PLE table is model-level and its conv is mirrored, so every device runs the whole conv and needs the whole history
        if (std::regex_match(tensor_name, pattern_ple_r_cache)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        // standard attention
        if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_kv_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_q_bias) || std::regex_match(tensor_name, pattern_kv_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_qkv_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if ( std::regex_match(tensor_name, pattern_qkv_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_qk_norm)) {
            return get_tensor_config_impl(tensor->ne[1] == 1 ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_kv_cache) || std::regex_match(tensor_name, pattern_attn_sinks)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }
        if (std::regex_match(tensor_name, pattern_attn_out_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        if (std::regex_match(tensor_name, pattern_attn_gate_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta) ||
                std::regex_match(tensor_name, pattern_ssm_beta_alpha)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_r_cache) || std::regex_match(tensor_name, pattern_s_cache)) {
            if (ud->model->arch == LLM_ARCH_LFM2 || ud->model->arch == LLM_ARCH_LFM2MOE) {
                // the LFM2 shortconv block runs fully mirrored, so its conv state must be mirrored too
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED, "");
            }
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_conv1d)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_out_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_gate_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_up_bias) || std::regex_match(tensor_name, pattern_ffn_gate_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_exps_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_PARTIAL, "ffn_down_exps.weight");
        }

        // output
        if (std::regex_match(tensor_name, pattern_output_weight)) {
            if (is_dsv4) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            }
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
        }
        if (std::regex_match(tensor_name, pattern_output_bias)) {
            const ggml_tensor * output_weight = ud->model->get_tensor("output.weight");
            GGML_ASSERT(output_weight != nullptr);
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }

        // everything else
        return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    };

    auto get_split_segments = [&](int axis, uint32_t il) -> std::vector<std::pair<int64_t, uint32_t>> {
        // TODO: clarify why this is necessary specifically for these models
        // TODO: deduplicate condition [TAG_SPLIT_QGATE_QWEN]
        if (ud->model->arch == LLM_ARCH_QWEN3NEXT || ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE ||
                ud->model->arch == LLM_ARCH_QWEN4EXP) {

            // fused full attention layers with Q gate tensors that need n_embd doubled:
            if (!hparams.is_recr(il) && (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias))) {
                const int64_t n_embd      = hparams.n_head(il) * hparams.n_embd_head_k(il) * 2;
                const int64_t n_embd_gqa  = hparams.n_embd_v_gqa(il);
                GGML_ASSERT(hparams.n_embd_k_gqa(il) == n_embd_gqa);
                GGML_ASSERT(tensor->ne[axis] == n_embd + 2*n_embd_gqa);
                return {{n_embd, 1}, {n_embd_gqa, 2}};
            }

            const int64_t head_k_dim = hparams.ssm_d_state;
            const int64_t head_v_dim = hparams.ssm_d_state;
            const int64_t n_k_heads  = hparams.ssm_n_group;
            const int64_t n_v_heads  = hparams.ssm_dt_rank;
            const int64_t key_dim    = head_k_dim * n_k_heads;
            const int64_t value_dim  = head_v_dim * n_v_heads;

            // both Qwen 3 Next and Qwen 3.5 support n_v_heads > n_k_heads but the broadcasting pattern is different:
            //   - Qwen 3 Next: [k0_v0, k0_v1, k1_v2, k1_v3] (this is the default split pattern)
            //   - Qwen 3.5:    [k0_v0, k1_v1, k0_v2, k1_v3] (needs segmenting of V on the scale of K to get the correct pattern)
            if (ud->model->arch == LLM_ARCH_QWEN3NEXT) {
                if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_ssm_conv1d)) {
                    GGML_ASSERT(tensor->ne[axis] == 2*key_dim + value_dim);
                    return {{key_dim, 2}, {value_dim, 1}};
                }
                if (std::regex_match(tensor_name, pattern_r_cache)) {
                    return {{key_dim * (hparams.ssm_d_conv - 1), 2}, {value_dim * (hparams.ssm_d_conv - 1), 1}};
                }
            } else {
                const int64_t head_ratio = n_v_heads / n_k_heads;
                if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_ssm_conv1d)) {
                    GGML_ASSERT(tensor->ne[axis] == 2*key_dim + value_dim);
                    return {{key_dim, 2 + head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_attn_gate_weight) || std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                    return {{key_dim, head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                        std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                    return {{n_k_heads, head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_r_cache)) {
                    return {{key_dim * (hparams.ssm_d_conv - 1), 2 + head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_s_cache)) {
                    return {{n_k_heads * head_v_dim * head_v_dim, head_ratio}};
                }
            }

            // the FFN is the same for Qwen 3 Next and Qwen 3.5:
            if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
                const int64_t n_ff_exp = hparams.n_ff_exp(il);
                GGML_ASSERT(tensor->ne[axis] == 2*n_ff_exp);
                return {{n_ff_exp, 2}};
            }
            return {{tensor->ne[axis], 1}};
        }

        if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
            const int64_t n_embd      = hparams.n_head(il) * hparams.n_embd_head_k(il);
            const int64_t n_embd_gqa  = hparams.n_embd_v_gqa(il);
            GGML_ASSERT(hparams.n_embd_k_gqa(il) == n_embd_gqa);
            GGML_ASSERT(tensor->ne[axis] == n_embd + 2*n_embd_gqa);
            return {{n_embd, 1}, {n_embd_gqa, 2}};
        }
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_up_bias)) {
            const int64_t n_ff = hparams.n_ff(il);
            // some models such as Phi 3 have fused up + gate tensors named "up" tensors, which need to be segmented
            if (tensor->ne[axis] == 2*n_ff) {
                return {{n_ff, 2}};
            }
            return {{tensor->ne[axis], 1}};
        }
        if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            const int64_t n_ff_exp = hparams.n_ff_exp(il);
            GGML_ASSERT(tensor->ne[axis] == 2*n_ff_exp);
            return {{n_ff_exp, 2}};
        }
        return {{tensor->ne[axis], 1}};
    };

    auto get_split_granularity = [&](int64_t blck_size, uint32_t il, const std::vector<std::pair<int64_t, uint32_t>> & segments) -> std::vector<int64_t> {
        // for better performance it may make sense to round up blck_size to a higher power of 2 so that more efficient kernels can be used
        if (hparams.is_recr(il)) {
            // linear attention
            const int64_t head_dim        = hparams.ssm_d_state;
            const int64_t blck_size_perf  = std::lcm(blck_size, 128);
            const int64_t granularity_qkv = std::lcm(blck_size_perf, head_dim);
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_attn_gate_weight) ||
                    std::regex_match(tensor_name, pattern_ssm_conv1d) || std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv);
            }
            if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                    std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv / head_dim);
            }
            if (std::regex_match(tensor_name, pattern_ssm_beta_alpha)) {
                return std::vector<int64_t>(segments.size(), 2 * (granularity_qkv / head_dim));
            }
            if (std::regex_match(tensor_name, pattern_r_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * (hparams.ssm_d_conv - 1));
            }
            if (std::regex_match(tensor_name, pattern_s_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * head_dim);
            }
        } else {
            // regular attention
            const uint32_t n_gqa    = hparams.n_gqa(il);
            const uint32_t n_embd_q = n_gqa * hparams.n_embd_head_k(il);

            // to handle head sizes like 80, only increase granularity while it doesn't cause underutilization
            int64_t blck_size_perf = blck_size;
            while (blck_size_perf < 128 && blck_size_perf*ud->n_devices < n_embd_q) {
                blck_size_perf *= 2;
            }

            const int64_t granularity_q    = std::lcm(n_embd_q, blck_size_perf);
            const int64_t granularity_head = granularity_q / hparams.n_embd_head_k(il); // for tensors with one value per head
            if (std::regex_match(tensor_name, pattern_attn_sinks)) {
                GGML_ASSERT(segments.size() == 1);
                if (is_dsv4) {
                    return {hparams.n_head(il) / hparams.dsv4_o_group_count};
                }
                return {granularity_head};
            }

            if (is_dsv4) {
                if (std::regex_match(tensor_name, pattern_attn_q_b_weight)) {
                    GGML_ASSERT(segments.size() == 1);
                    // the grouped output projection requires each device to hold whole groups of heads
                    const int64_t n_head_group = hparams.n_head(il) / hparams.dsv4_o_group_count;
                    return {n_head_group * hparams.n_embd_head_k(il)};
                }
                if (std::regex_match(tensor_name, pattern_attn_out_a_weight)) {
                    GGML_ASSERT(segments.size() == 1);
                    return {1};
                }
                if (std::regex_match(tensor_name, pattern_attn_out_b_weight)) {
                    GGML_ASSERT(segments.size() == 1);
                    // the boundaries must align with wo_a's per-group split, so quant blocks must not straddle groups
                    GGML_ASSERT(hparams.dsv4_o_lora_rank % blck_size == 0);
                    return {hparams.dsv4_o_lora_rank};
                }
            }
            if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_q_bias)) {
                GGML_ASSERT(segments.size() == 1);
                // some models have Q gate tensors, for those cases the granularity needs to be doubled:
                // TODO: deduplicate condition [TAG_SPLIT_QGATE_QWEN]
                if (ud->model->arch == LLM_ARCH_QWEN3NEXT || ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE ||
                        ud->model->arch == LLM_ARCH_QWEN4EXP) {
                    return {std::lcm(2*n_embd_q, blck_size_perf)};
                }
                return {granularity_q};
            }
            if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {granularity_q};
            }
            if (std::regex_match(tensor_name, pattern_attn_gate_weight)) {
                GGML_ASSERT(segments.size() == 1);
                if (tensor->ne[1] == hparams.n_head(il)) {
                    return {granularity_head};
                }
                return {granularity_q};
            }

            const int64_t granularity_kv = granularity_q / n_gqa;
            if (std::regex_match(tensor_name, pattern_kv_weight) ||
                std::regex_match(tensor_name, pattern_kv_bias) ||
                std::regex_match(tensor_name, pattern_kv_cache)) {
                GGML_ASSERT(segments.size() == 1);
                return {granularity_kv};
            }
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
                GGML_ASSERT(segments.size() == 2);
                // fused full attention layers need Q gate tensors handled like above:
                // TODO: deduplicate condition [TAG_SPLIT_QGATE_QWEN]
                if (ud->model->arch == LLM_ARCH_QWEN3NEXT || ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE ||
                        ud->model->arch == LLM_ARCH_QWEN4EXP) {
                    return {std::lcm(2*n_embd_q, blck_size_perf), granularity_kv};
                }
                return {granularity_q, granularity_kv};
            }
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_up_bias) ||
                std::regex_match(tensor_name, pattern_ffn_gate_weight) || std::regex_match(tensor_name, pattern_ffn_gate_bias) ||
                std::regex_match(tensor_name, pattern_ffn_gate_up_weight) ||
                std::regex_match(tensor_name, pattern_ffn_down_weight) ||
                std::regex_match(tensor_name, pattern_ffn_up_shexp_weight) ||
                std::regex_match(tensor_name, pattern_ffn_gate_shexp_weight) ||
                std::regex_match(tensor_name, pattern_ffn_down_shexp_weight)) {
            const int64_t blck_size_perf = std::lcm(blck_size, 128);
            GGML_ASSERT(segments.size() == 1);
            return {blck_size_perf};
        }

        // everything else
        GGML_ASSERT(segments.size() == 1);
        return {1};
    };

    ggml_backend_meta_split_state split_state;
    memset(&split_state, 0, sizeof(split_state));
    tensor_config tc = get_tensor_config();
    split_state.axis = tc.axis;
    if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
        const int64_t blck_size = ggml_blck_size(tc.tensor_axis_0->type);
        const float * tensor_split = ud->model->tensor_split();
        std::vector<float> tensor_split_scan;
        tensor_split_scan.reserve(ud->n_devices);
        for (size_t j = 0; j < ud->n_devices; j++) {
            tensor_split_scan.push_back(tensor_split == nullptr ? 0.0f : tensor_split[(j + tc.rotation) % ud->n_devices]);
            if (j > 0) {
                tensor_split_scan[j] += tensor_split_scan[j - 1];
            }
        }
        const std::vector<std::pair<int64_t, uint32_t>> segments = get_split_segments(split_state.axis, tc.il);
        const std::vector<int64_t> granularity = get_split_granularity(blck_size, tc.il, segments);
        for (size_t is = 0; is < segments.size(); is++) {
            const int64_t  ne_s = segments[is].first;
            const uint32_t nr_s = segments[is].second;
            const int64_t  g_s  = granularity[is];
            int64_t low = 0;
            size_t j = 0;
            for (; j < ud->n_devices - 1; j++) {
                int64_t high = tensor_split_scan.back() == 0.0f ?
                    ne_s * (j+1)/ud->n_devices : ne_s * tensor_split_scan[j]/tensor_split_scan.back();
                if (high % g_s != 0) {
                    high -= high % g_s;
                }
                split_state.ne[is*ud->n_devices + (j + tc.rotation) % ud->n_devices] = high - low;
                low = high;
            }
            split_state.ne[is*ud->n_devices + (j + tc.rotation) % ud->n_devices] = ne_s - low;
            split_state.nr[is] = nr_s;
        }
        split_state.n_segments = segments.size();
    } else {
        memset(split_state.ne, 0, sizeof(split_state.ne));
        split_state.nr[0] = 1;
        split_state.n_segments = 1;
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            GGML_ASSERT(tc.tensor_axis_0 != tensor);
            const ggml_backend_meta_split_state source_split_state = llama_meta_device_get_split_state(tc.tensor_axis_0, userdata);
            GGML_ASSERT(source_split_state.axis >= 0 && source_split_state.axis < GGML_MAX_DIMS);
            for (size_t j = 0; j < ud->n_devices; j++) {
                for (size_t is = 0; is < source_split_state.n_segments; is++) {
                    split_state.ne[j] += source_split_state.ne[is*ud->n_devices + j] * source_split_state.nr[is];
                }
            }
        }
    }
    return split_state;
    GGML_UNUSED(userdata);
}

const char * llm_type_name(llm_type type) {
    switch (type) {
        case LLM_TYPE_14M:           return "14M";
        case LLM_TYPE_17M:           return "17M";
        case LLM_TYPE_22M:           return "22M";
        case LLM_TYPE_33M:           return "33M";
        case LLM_TYPE_47M:           return "47M";
        case LLM_TYPE_60M:           return "60M";
        case LLM_TYPE_70M:           return "70M";
        case LLM_TYPE_80M:           return "80M";
        case LLM_TYPE_109M:          return "109M";
        case LLM_TYPE_137M:          return "137M";
        case LLM_TYPE_140M:          return "140M";
        case LLM_TYPE_149M:          return "149M";
        case LLM_TYPE_160M:          return "160M";
        case LLM_TYPE_190M:          return "190M";
        case LLM_TYPE_220M:          return "220M";
        case LLM_TYPE_230M:          return "230M";
        case LLM_TYPE_250M:          return "250M";
        case LLM_TYPE_256M:          return "256M";
        case LLM_TYPE_270M:          return "270M";
        case LLM_TYPE_335M:          return "335M";
        case LLM_TYPE_350M:          return "350M";
        case LLM_TYPE_360M:          return "360M";
        case LLM_TYPE_395M:          return "395M";
        case LLM_TYPE_410M:          return "410M";
        case LLM_TYPE_450M:          return "450M";
        case LLM_TYPE_475M:          return "475M";
        case LLM_TYPE_558M:          return "558M";
        case LLM_TYPE_700M:          return "700M";
        case LLM_TYPE_770M:          return "770M";
        case LLM_TYPE_780M:          return "780M";
        case LLM_TYPE_950M:          return "950M";
        case LLM_TYPE_0_3B:          return "0.3B";
        case LLM_TYPE_0_5B:          return "0.5B";
        case LLM_TYPE_0_6B:          return "0.6B";
        case LLM_TYPE_0_8B:          return "0.8B";
        case LLM_TYPE_1B:            return "1B";
        case LLM_TYPE_1_2B:          return "1.2B";
        case LLM_TYPE_1_3B:          return "1.3B";
        case LLM_TYPE_1_4B:          return "1.4B";
        case LLM_TYPE_1_5B:          return "1.5B";
        case LLM_TYPE_1_6B:          return "1.6B";
        case LLM_TYPE_1_7B:          return "1.7B";
        case LLM_TYPE_1_8B:          return "1.8B";
        case LLM_TYPE_2B:            return "2B";
        case LLM_TYPE_2_6B:          return "2.6B";
        case LLM_TYPE_2_8B:          return "2.8B";
        case LLM_TYPE_2_9B:          return "2.9B";
        case LLM_TYPE_3B:            return "3B";
        case LLM_TYPE_4B:            return "4B";
        case LLM_TYPE_6B:            return "6B";
        case LLM_TYPE_6_9B:          return "6.9B";
        case LLM_TYPE_7B:            return "7B";
        case LLM_TYPE_8B:            return "8B";
        case LLM_TYPE_9B:            return "9B";
        case LLM_TYPE_11B:           return "11B";
        case LLM_TYPE_12B:           return "12B";
        case LLM_TYPE_13B:           return "13B";
        case LLM_TYPE_14B:           return "14B";
        case LLM_TYPE_15B:           return "15B";
        case LLM_TYPE_16B:           return "16B";
        case LLM_TYPE_20B:           return "20B";
        case LLM_TYPE_26B:           return "26B";
        case LLM_TYPE_27B:           return "27B";
        case LLM_TYPE_30B:           return "30B";
        case LLM_TYPE_31B:           return "31B";
        case LLM_TYPE_32B:           return "32B";
        case LLM_TYPE_34B:           return "34B";
        case LLM_TYPE_35B:           return "35B";
        case LLM_TYPE_36B:           return "36B";
        case LLM_TYPE_40B:           return "40B";
        case LLM_TYPE_65B:           return "65B";
        case LLM_TYPE_70B:           return "70B";
        case LLM_TYPE_120B:          return "120B";
        case LLM_TYPE_142B:          return "142B";
        case LLM_TYPE_236B:          return "236B";
        case LLM_TYPE_290B:          return "290B";
        case LLM_TYPE_314B:          return "314B";
        case LLM_TYPE_405B:          return "405B";
        case LLM_TYPE_456B:          return "456B";
        case LLM_TYPE_671B:          return "671B";
        case LLM_TYPE_SMALL:         return "0.1B";
        case LLM_TYPE_MEDIUM:        return "0.4B";
        case LLM_TYPE_LARGE:         return "0.8B";
        case LLM_TYPE_XL:            return "1.5B";
        case LLM_TYPE_A1_7B:         return "A1.7B";
        case LLM_TYPE_A2_7B:         return "A2.7B";
        case LLM_TYPE_8x7B:          return "8x7B";
        case LLM_TYPE_8x22B:         return "8x22B";
        case LLM_TYPE_16x12B:        return "16x12B";
        case LLM_TYPE_16x3_8B:       return "16x3.8B";
        case LLM_TYPE_10B_128x3_66B: return "10B+128x3.66B";
        case LLM_TYPE_57B_A14B:      return "57B.A14B";
        case LLM_TYPE_17B_16E:       return "17Bx16E (Scout)";
        case LLM_TYPE_17B_128E:      return "17Bx128E (Maverick)";
        case LLM_TYPE_A13B:          return "A13B";
        case LLM_TYPE_1B_A400M:      return "1B.A400M";
        case LLM_TYPE_3B_A800M:      return "3B.A800M";
        case LLM_TYPE_7B_A1B:        return "7B.A1B";
        case LLM_TYPE_8B_A1B:        return "8B.A1B";
        case LLM_TYPE_7_9B_A1_3B:    return "7.9B.A1.3B";
        case LLM_TYPE_12B_A2_5B:     return "12B.A2.5B";
        case LLM_TYPE_16B_A1B:       return "16B.A1B";
        case LLM_TYPE_21B_A3B:       return "21B.A3B";
        case LLM_TYPE_24B_A2B:       return "24B.A2B";
        case LLM_TYPE_26B_A4B:       return "26B.A4B";
        case LLM_TYPE_30B_A3B:       return "30B.A3B";
        case LLM_TYPE_31B_A3_5B:     return "31B.A3.5B";
        case LLM_TYPE_32B_A9B:       return "32B.A9B";
        case LLM_TYPE_35B_A3B:       return "35B.A3B";
        case LLM_TYPE_48B_A3B:       return "48B.A3B";
        case LLM_TYPE_75B_A9B:       return "75B.A9B";
        case LLM_TYPE_80B_A3B:       return "80B.A3B";
        case LLM_TYPE_A3B:           return "A3B";
        case LLM_TYPE_100B_A6B:      return "100B.A6B";
        case LLM_TYPE_102B_A12B:     return "102B.A12B";
        case LLM_TYPE_106B_A12B:     return "106B.A12B";
        case LLM_TYPE_118B_A8B:      return "118B.A8B";
        case LLM_TYPE_120B_A12B:     return "120B.A12B";
        case LLM_TYPE_122B_A10B:     return "122B.A10B";
        case LLM_TYPE_124B_A5_1B:    return "124B.A5.1B";
        case LLM_TYPE_196B_A11B:     return "196B.A11B";
        case LLM_TYPE_230B_A10B:     return "230B.A10B";
        case LLM_TYPE_428B_A23B:     return "428B.A23B";
        case LLM_TYPE_235B_A22B:     return "235B.A22B";
        case LLM_TYPE_288B_A19B:     return "288B.A19B";
        case LLM_TYPE_300B_A47B:     return "300B.A47B";
        case LLM_TYPE_310B_A15B:     return "310B.A15B";
        case LLM_TYPE_355B_A32B:     return "355B.A32B";
        case LLM_TYPE_397B_A17B:     return "397B.A17B";
        case LLM_TYPE_685B_A37B:     return "685B.A37B";
        case LLM_TYPE_744B_A40B:     return "744B.A40B";
        case LLM_TYPE_2_8T_A50B:     return "2.8T.A50B";
        case LLM_TYPE_E2B:           return "E2B";
        case LLM_TYPE_E4B:           return "E4B";
        default:                     return "?B";
    }
}

static const char * llama_expert_gating_func_name(llama_expert_gating_func_type type) {
    switch (type) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX: return "softmax";
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID: return "sigmoid";
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS: return "sqrtsoftplus";
        default:                                    return "unknown";
    }
}

static const std::map<llama_rope_scaling_type, const char *> LLAMA_ROPE_SCALING_TYPES = {
    { LLAMA_ROPE_SCALING_TYPE_NONE,       "none"       },
    { LLAMA_ROPE_SCALING_TYPE_LINEAR,     "linear"     },
    { LLAMA_ROPE_SCALING_TYPE_YARN,       "yarn"       },
    { LLAMA_ROPE_SCALING_TYPE_LONGROPE,   "longrope"   },
};

std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type) {
    return LLAMA_ROPE_SCALING_TYPES.at(rope_scaling_type);
}

static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (llama_rope_scaling_type) kv.first;
        }
    }

    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

// Maps the GGUF `<arch>.hidden_activation` string to the FFN op type used by the
// graph builders. Only gated activations that map cleanly to llm_ffn_op_type are
// listed; unrecognized values fall back to GeGLU, which matches the historical
// default for ModernBert-style architectures.
static const std::map<std::string, llm_ffn_op_type> LLM_FFN_OP_TYPES_FROM_STRING = {
    { "gelu",   LLM_FFN_GEGLU  },
    { "geglu",  LLM_FFN_GEGLU  },
    { "silu",   LLM_FFN_SWIGLU },
    { "swish",  LLM_FFN_SWIGLU },
    { "swiglu", LLM_FFN_SWIGLU },
    { "relu",   LLM_FFN_RELU   },
    { "reglu",  LLM_FFN_REGLU  },
};

llm_ffn_op_type llm_ffn_op_type_from_string(const std::string & name, llm_ffn_op_type fallback) {
    const auto it = LLM_FFN_OP_TYPES_FROM_STRING.find(name);
    if (it != LLM_FFN_OP_TYPES_FROM_STRING.end()) {
        return it->second;
    }
    return fallback;
}

// CPU: ACCEL -> GPU host -> CPU extra -> CPU
static buft_list_t make_cpu_buft_list(const std::vector<llama_device> & devices, bool use_extra_bufts, bool no_host) {
    buft_list_t buft_list;

    // add ACCEL buffer types
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            auto * buft = ggml_backend_dev_buffer_type(dev);
            // skip
            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    // generally, this will be done using the first device in the list
    // a better approach would be to handle this on a weight-by-weight basis using the offload_op
    // function of the device to determine if it would benefit from being stored in a host buffer
    if (!no_host) {
        for (const auto & dev : devices) {
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev.dev);
            if (buft) {
                buft_list.emplace_back(dev.dev, buft);
                break;
            }
        }
    }

    // add extra buffer types
    if (use_extra_bufts) {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }

        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(cpu_dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // add the CPU buffer type
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
        }
    }

    return buft_list;
}

// GPU: split if LLAMA_SPLIT_MODE_ROW -> GPU
static buft_list_t make_gpu_buft_list(ggml_backend_dev_t dev, llama_split_mode split_mode, const float * tensor_split) {
    buft_list_t buft_list;

    // add the device split buffer type if requested and available
    if (split_mode == LLAMA_SPLIT_MODE_ROW) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_split_buffer_type_fn = (ggml_backend_split_buffer_type_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
        if (ggml_backend_split_buffer_type_fn) {
            size_t dev_index = [&]() {
                auto * reg = ggml_backend_dev_backend_reg(dev);
                for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
                    if (ggml_backend_reg_dev_get(reg, i) == dev) {
                        return i;
                    }
                }
                throw std::runtime_error(format("device %s not found in its backend reg", ggml_backend_dev_name(dev)));
            }();
            auto * buft = ggml_backend_split_buffer_type_fn(dev_index, tensor_split);
            if (buft != nullptr) {
                buft_list.emplace_back(dev, buft);
            }
        } else {
            throw std::runtime_error(format("device %s does not support split buffers", ggml_backend_dev_name(dev)));
        }
    }

    // add the device default buffer type
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));

    // add the device extra buffer type (if any)
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg) {
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");

        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    return buft_list;
}

struct llama_model::impl {
    impl() = default;
    ~impl() = default;

    uint64_t n_elements = 0;

    size_t n_bytes = 0;

    std::string desc_str;

    llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

    // model memory mapped files
    llama_mmaps mappings;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;

    // contexts where the model tensors metadata is stored as well as the corresponding buffers:
    std::vector<std::pair<ggml_context_ptr, std::vector<ggml_backend_buffer_ptr>>> ctxs_bufs;
    std::vector<llama_moe_source_group> moe_sources;

    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct layer_dev {
        ggml_backend_dev_t dev;
        buft_list_t * buft_list;
    };

    layer_dev dev_input = {};
    layer_dev dev_output = {};
    std::vector<layer_dev> dev_layer;

    bool has_tensor_overrides;

    std::vector<float> tensor_split_owned;
    struct tensor_override_resolution {
        uint32_t origin = llama_model_loader::TENSOR_OVERRIDE_DEFAULT;
        int32_t index = -1;
        std::string pattern;
        std::string requested_buft;
        std::string selected_buft;
        std::string resolved_buft;
        std::string resolved_owner_canonical_id;
        std::string resolved_owner_identity_kind;
        std::string resolved_owner_backend;
        bool        resolved_is_host = false;
    };
    std::unordered_map<std::string, tensor_override_resolution> tensor_override_resolutions;
    std::unordered_map<std::string, llama_moe_shared_tensor> borrowed_tensors;
    struct artifact_source {
        size_t  file_size = 0;
        int64_t modification_time = 0;
        bool    modification_time_available = false;
    };
    std::vector<artifact_source> artifact_sources;
    bool resolved_uses_mmap = false;
    std::string resolved_direct_io_state = "none";
    bool resolved_uses_mlock = false;
    bool resolved_has_lazy_tensors = false;
    std::vector<llama_model_layer_range> moe_cache_layer_ranges_owned;
    std::vector<size_t> moe_cache_byte_budgets_owned;
    std::unordered_map<const ggml_tensor *, ggml_backend_dev_t> moe_cache_tensor_owners;
    std::map<ggml_backend_dev_t, llama_moe_cache_memory> moe_cache_memory;
    std::array<std::vector<llama_moe_cache_memory>, 2>                  moe_cache_group_context_memory;
    std::array<std::map<ggml_backend_dev_t, llama_moe_cache_memory>, 2> moe_cache_context_memory;
    std::array<size_t, 2>                                               moe_cache_context_host_memory = {};
    struct moe_cache_staging_group {
        ggml_backend_dev_t owner;
        int32_t layer;
        uint32_t n_experts;
        uint32_t top_k;
        uint32_t context_mask;
        std::vector<uint64_t> bank_expert_strides;
    };
    std::vector<moe_cache_staging_group> moe_cache_staging_groups;
    std::map<ggml_backend_dev_t, int32_t> moe_cache_slots;
};

llama_model::llama_model(const llama_model_params & params) : params(params), pimpl(std::make_unique<impl>()) {
    if ((params.moe_expert_cache_byte_budgets == nullptr) != (params.n_moe_expert_cache_byte_budgets == 0) ||
            (params.moe_expert_cache_layer_ranges == nullptr) != (params.n_moe_expert_cache_layer_ranges == 0)) {
        throw std::runtime_error("invalid MoE expert cache parameter array");
    }
    if (params.n_moe_expert_cache_byte_budgets != 0 && params.moe_expert_cache_slots > 0) {
        throw std::runtime_error("MoE cache byte budgets conflict with a nonzero slot count");
    }
    if (params.tensor_split != nullptr) {
        // llama_model_params stores tensor_split as a borrowed pointer, but the model
        // may need it later for tensor-parallel KV-cache split metadata.
        pimpl->tensor_split_owned.assign(params.tensor_split, params.tensor_split + llama_max_devices());
        this->params.tensor_split = pimpl->tensor_split_owned.data();
    }
    if (params.moe_expert_cache_layer_ranges != nullptr && params.n_moe_expert_cache_layer_ranges != 0) {
        pimpl->moe_cache_layer_ranges_owned.assign(
            params.moe_expert_cache_layer_ranges,
            params.moe_expert_cache_layer_ranges + params.n_moe_expert_cache_layer_ranges);
        this->params.moe_expert_cache_layer_ranges = pimpl->moe_cache_layer_ranges_owned.data();
    }
    if (params.moe_expert_cache_byte_budgets != nullptr && params.n_moe_expert_cache_byte_budgets != 0) {
        pimpl->moe_cache_byte_budgets_owned.assign(
            params.moe_expert_cache_byte_budgets,
            params.moe_expert_cache_byte_budgets + params.n_moe_expert_cache_byte_budgets);
        this->params.moe_expert_cache_byte_budgets = pimpl->moe_cache_byte_budgets_owned.data();
        // Budget mode uses owner-local slot counts. Keep a positive sentinel for
        // existing cache-enabled gates; candidate snapshots replace it per owner.
        this->params.moe_expert_cache_slots = 1;
    }
    pimpl->has_tensor_overrides = params.tensor_buft_overrides && params.tensor_buft_overrides[0].pattern;
}

llama_model::~llama_model() {
    for (auto * lora : loras) {
        delete lora;
    }
}

void llama_model_base::load_stats(llama_model_loader & ml) {
    pimpl->n_elements = ml.n_elements;
    pimpl->n_bytes = ml.n_bytes;
}

void llama_model_base::load_hparams(llama_model_loader & ml) {
    const gguf_context * ctx = ml.metadata;

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char * name = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        gguf_kv.emplace(name, value);
    }

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, name, false);

    // everything past this point is not vocab-related
    // for CLIP models, we only need to load tensors, no hparams
    if (hparams.vocab_only || ml.get_arch() == LLM_ARCH_CLIP) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH,          hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,        hparams.n_embd);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_OUT,    hparams.n_embd_out_impl, false);
    ml.get_key(LLM_KV_ATTENTION_CAUSAL,        hparams.causal_attn,     false);
    ml.get_key(LLM_KV_POOLING_TYPE,            hparams.pooling_type,    false);
    ml.get_key(LLM_KV_BLOCK_COUNT,             hparams.n_layer_all);
    GGML_ASSERT(hparams.n_layer_all > 0 && hparams.n_layer_all <= LLAMA_MAX_LAYERS);
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,    hparams.n_layer_nextn,   false);
    GGML_ASSERT(hparams.n_layer_nextn <= hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_COUNT,            hparams.n_expert,        false);
    std::fill(hparams.n_expert_used_arr.begin(), hparams.n_expert_used_arr.end(), 0);
    ml.get_key_or_arr(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,      hparams.n_expert_groups, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT, hparams.n_group_used,    false);

    if (arch == LLM_ARCH_HUNYUAN_VL || arch == LLM_ARCH_HUNYUAN_DENSE) {
        if (hparams.n_expert <= 1) {
            hparams.n_expert = 0;
            std::fill(hparams.n_expert_used_arr.begin(), hparams.n_expert_used_arr.end(), 0);
        }
    }

    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        ml.get_key(LLM_KV_FEATURES_LENGTH,  hparams.n_embd);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH, hparams.n_embd_out_impl);

        ml.get_key(LLM_KV_POSNET_EMBEDDING_LENGTH, hparams.posnet.n_embd);
        ml.get_key(LLM_KV_POSNET_BLOCK_COUNT,      hparams.posnet.n_layer);

        ml.get_key(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, hparams.convnext.n_embd);
        ml.get_key(LLM_KV_CONVNEXT_BLOCK_COUNT,      hparams.convnext.n_layer);

        GGML_ASSERT(hparams.posnet.n_layer   <= hparams.n_layer_all);
        GGML_ASSERT(hparams.convnext.n_layer <= hparams.n_layer_all);
    }

    // models may route a different number of experts per layer, so validate the maximum
    uint32_t n_expert_used_max = hparams.n_expert_used_max();

    GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
    GGML_ASSERT(n_expert_used_max <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(n_expert_used_max > 0);
        GGML_ASSERT(hparams.n_expert_groups < hparams.n_expert);
        if (hparams.n_expert_groups > 1) {
            GGML_ASSERT(hparams.n_expert % hparams.n_expert_groups == 0);
            GGML_ASSERT(hparams.n_group_used > 0);
            GGML_ASSERT(hparams.n_group_used < hparams.n_expert_groups);
        }
    } else {
        GGML_ASSERT(n_expert_used_max == 0);
        GGML_ASSERT(hparams.n_expert_groups == 0);
    }

    std::fill(hparams.n_head_arr.begin(),       hparams.n_head_arr.end(),       0);
    std::fill(hparams.n_head_kv_arr.begin(),    hparams.n_head_kv_arr.end(),    0);
    std::fill(hparams.n_ff_arr.begin(),         hparams.n_ff_arr.end(),         0);
    std::fill(hparams.n_ff_exp_arr.begin(),     hparams.n_ff_exp_arr.end(),     0);

    std::fill(hparams.rope_sections.begin(), hparams.rope_sections.end(), 0);
    std::fill(hparams.rope_pattern.begin(),  hparams.rope_pattern.end(), 1);
    std::fill(hparams.is_swa_impl.begin(),   hparams.is_swa_impl.end(), 0);
    std::fill(hparams.is_recr_impl.begin(),  hparams.is_recr_impl.end(),  llm_arch_is_recurrent(ml.get_arch()) ? 1 : 0);
    std::fill(hparams.is_indexer_full_impl.begin(), hparams.is_indexer_full_impl.end(), 0);

    std::fill(hparams.xielu_alpha_n.begin(), hparams.xielu_alpha_n.end(), 0.0f);
    std::fill(hparams.xielu_alpha_p.begin(), hparams.xielu_alpha_p.end(), 0.0f);
    std::fill(hparams.xielu_beta.begin(),    hparams.xielu_beta.end(), 0.0f);
    std::fill(hparams.xielu_eps.begin(),     hparams.xielu_eps.end(), 0.0f);

    std::fill(hparams.swiglu_clamp_exp.begin(),   hparams.swiglu_clamp_exp.end(),   0.0f);
    std::fill(hparams.swiglu_clamp_shexp.begin(), hparams.swiglu_clamp_shexp.end(), 0.0f);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer_all, false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer_all, false);

    // Populate deepstack_mapping_arr - initialized to -1 (no deepstack)
    std::fill(hparams.deepstack_mapping_arr.begin(), hparams.deepstack_mapping_arr.end(), -1);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer_all, false);

    bool rope_finetuned = false;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
    GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);

    // TODO: Handle SWA metadata similarly when models start implementing it
    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;

    ml.get_key(LLM_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);
    ml.get_key(LLM_KV_ROPE_SCALING_ALPHA,       hparams.rope_scaling_alpha, false);

    // non-transformer models do not have attention heads
    if (hparams.n_head() > 0) {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        hparams.n_embd_head_k_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k_full, false);

        hparams.n_embd_head_v_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v_full, false);

        // sanity check for n_rot (optional)
        hparams.n_rot_full = hparams.n_embd_head_k_full;

        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot_full, false);

        if (arch == LLM_ARCH_LLAMA || arch == LLM_ARCH_DECI || arch == LLM_ARCH_FALCON || arch == LLM_ARCH_LLAMA_EMBED) {
            if (hparams.n_rot_full != hparams.n_embd_head_k_full) {
                throw std::runtime_error(format("invalid n_rot: %u, expected %u", hparams.n_rot_full, hparams.n_embd_head_k_full));
            }
        }
    } else {
        hparams.n_rot_full = 0;
        hparams.n_embd_head_k_full = 0;
        hparams.n_embd_head_v_full = 0;
    }

    // head size and n_rot for SWA layers
    {
        hparams.n_embd_head_k_swa = hparams.n_embd_head_k_full;
        hparams.n_embd_head_v_swa = hparams.n_embd_head_v_full;
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA, hparams.n_embd_head_k_swa, false);
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA, hparams.n_embd_head_v_swa, false);

        hparams.n_rot_swa = hparams.n_rot_full;
        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT_SWA, hparams.n_rot_swa, false);
    }

    // for classifier models
    ml.get_arr(LLM_KV_CLASSIFIER_OUTPUT_LABELS, classifier_labels, false);
    if (!classifier_labels.empty()) {
        hparams.n_cls_out = classifier_labels.size();
    }

    // per-arch hparams
    load_arch_hparams(ml);

    pimpl->n_bytes = ml.n_bytes;

    pimpl->desc_str = arch_name() + " " + type_name() + " " + ml.ftype_name();

    pimpl->ftype = ml.ftype;

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.use_alibi = true;
    }

    hparams.rope_type = llama_model_rope_type(this);
}

void llama_model_base::load_vocab(llama_model_loader & ml) {
    const auto kv = LLM_KV(arch);

    vocab.load(ml, kv);
}

bool llama_model_base::load_tensors(llama_model_loader & ml) {
    const auto & split_mode   = params.split_mode;
    const bool use_mlock      = params.load_mode == LLAMA_LOAD_MODE_MLOCK || params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
    const auto & tensor_split = params.tensor_split;

    const int n_layer_all = hparams.n_layer_all;
    const int n_gpu_layers = this->n_gpu_layers();

    const bool use_mmap_buffer = true;

    this->ml = &ml; // to be used by create_tensor() and load_arch_tensors()

    if (ml.use_mmap && params.load_mode == LLAMA_LOAD_MODE_AUTO) {
        for (const auto & dev : devices) {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev.dev, &props);
            if (!props.caps.mmap_support) {
                ml.use_mmap = false;
                break;
            }
        }
    }

    // resolve AUTO on systems without mmap support (e.g. iGPUs): fall back to OFF; see #28160
    if (ml.lazy.mode == LLAMA_LAZY_MODE_AUTO) {
        for (const auto & dev : devices) {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev.dev, &props);
            if (!props.caps.mmap_support) {
                ml.lazy.mode = LLAMA_LAZY_MODE_OFF;
                break;
            }
        }
    }

    const char * load_mode_name = params.load_mode == LLAMA_LOAD_MODE_AUTO
        ? llama_load_mode_name(ml.use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE)
        : llama_load_mode_name(params.load_mode);

    LLAMA_LOG_INFO("%s: loading model tensors, this can take a while... (load_mode = %s)\n",
        __func__, load_mode_name);

    // build a list of buffer types for the CPU and GPU devices
    pimpl->cpu_buft_list = make_cpu_buft_list(devices, params.use_extra_bufts, params.no_host);
    for (const auto & dev : devices) {
        buft_list_t buft_list = make_gpu_buft_list(dev.dev, split_mode, tensor_split);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end());
        pimpl->gpu_buft_list.emplace(dev.dev, std::move(buft_list));
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        throw std::runtime_error(format("%s: no CPU backend found", __func__));
    }

    // calculate the split points
    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + n_devices(), [](float x) { return x == 0.0f; });
    std::vector<float> splits(n_devices());
    if (all_zero) {
        // default split, by free memory
        for (size_t i = 0; i < n_devices(); ++i) {
            ggml_backend_dev_t dev = devices[i].dev;
            size_t total;
            size_t free;
            ggml_backend_dev_memory(dev, &free, &total);

            // devices can return 0 bytes for free and total memory if they do not
            // have any to report. in this case, we will use the host memory as a fallback
            // fixes: https://github.com/ggml-org/llama.cpp/issues/18577
            if (free == 0 && total == 0) {
                ggml_backend_dev_memory(cpu_dev, &free, &total);
            }
            splits[i] = free;
        }
    } else {
        std::copy(tensor_split, tensor_split + n_devices(), splits.begin());
    }

    // sum and normalize the splits to get the split points
    float split_sum = 0.0f;
    for (size_t i = 0; i < n_devices(); ++i) {
        split_sum += splits[i];
        splits[i] = split_sum;
    }
    for (size_t i = 0; i < n_devices(); ++i) {
        splits[i] /= split_sum;
    }

    const int i_gpu_start = std::max(n_layer_all + 1 - n_gpu_layers, 0);
    const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, n_layer_all + 1);
    auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
        const bool is_swa = il < n_layer_all && hparams.is_swa(il);
        if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
            LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(cpu_dev), is_swa);
            return {cpu_dev, &pimpl->cpu_buft_list};
        }
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu).dev;
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};
    };

    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };

    // assign the repeating layers to the devices according to the splits
    pimpl->dev_layer.resize(n_layer_all);
    for (int il = 0; il < n_layer_all; ++il) {
        pimpl->dev_layer[il] = get_layer_buft_list(il);
    }

    // assign the output layer
    pimpl->dev_output = get_layer_buft_list(n_layer_all);

    const auto TENSOR_NOT_REQUIRED = llama_model_loader::TENSOR_NOT_REQUIRED;

    // create tensors for the weights
    {
        // TODO: move to a separate function
        const auto tn = LLM_TN(arch);

        const int64_t n_expert = hparams.n_expert;

        if (n_expert > 0 && hparams.n_expert_used_max() == 0) {
            throw std::runtime_error("model has expert layers but no expert layers are used");
        }

        layers.resize(n_layer_all);

        // call the per-model loading function
        load_arch_tensors(ml);

        // generic pass: load optional per-tensor/per-expert ".scale" tensors (e.g. NVFP4 scale2)
        // this avoids having to add scale loading to every architecture
        for (int i = 0; i < n_layer_all; ++i) {
            auto & layer = layers[i];

            // attention weight scales (per-tensor, shape {1})
            if (!layer.wq_s && layer.wq) {
                layer.wq_s = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wk_s && layer.wk) {
                layer.wk_s = create_tensor(tn(LLM_TENSOR_ATTN_K,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wv_s && layer.wv) {
                layer.wv_s = create_tensor(tn(LLM_TENSOR_ATTN_V,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wo_s && layer.wo) {
                layer.wo_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_s && layer.wqkv) {
                layer.wqkv_s = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_gate_s && layer.wqkv_gate) {
                layer.wqkv_gate_s = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // dense FFN weight scales (per-tensor, shape {1})
            if (!layer.ffn_gate_s && layer.ffn_gate) {
                layer.ffn_gate_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_s && layer.ffn_down) {
                layer.ffn_down_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_s && layer.ffn_up) {
                layer.ffn_up_s = create_tensor(tn(LLM_TENSOR_FFN_UP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_shexp_s && layer.ffn_gate_shexp) {
                layer.ffn_gate_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_shexp_s && layer.ffn_down_shexp) {
                layer.ffn_down_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_shexp_s && layer.ffn_up_shexp) {
                layer.ffn_up_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // MoE expert weight scales (per-expert, shape {n_expert})
            if (!layer.ffn_gate_exps_s && layer.ffn_gate_exps) {
                layer.ffn_gate_exps_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_exps_s && layer.ffn_down_exps) {
                layer.ffn_down_exps_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_exps_s && layer.ffn_up_exps) {
                layer.ffn_up_exps_s = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }

            // recurrent / linear-attention weight scales (per-tensor, shape {1})
            if (!layer.ssm_in_s && layer.ssm_in) {
                layer.ssm_in_s = create_tensor(tn(LLM_TENSOR_SSM_IN, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_out_s && layer.ssm_out) {
                layer.ssm_out_s = create_tensor(tn(LLM_TENSOR_SSM_OUT, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_alpha_s && layer.ssm_alpha) {
                layer.ssm_alpha_s = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_beta_s && layer.ssm_beta) {
                layer.ssm_beta_s = create_tensor(tn(LLM_TENSOR_SSM_BETA, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.eh_proj_s && layer.nextn.eh_proj) {
                layer.nextn.eh_proj_s = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.shared_head_head_s && layer.nextn.shared_head_head) {
                layer.nextn.shared_head_head_s = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // input scales
            if (!layer.wq_in_s && layer.wq) {
                layer.wq_in_s = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wk_in_s && layer.wk) {
                layer.wk_in_s = create_tensor(tn(LLM_TENSOR_ATTN_K,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wv_in_s && layer.wv) {
                layer.wv_in_s = create_tensor(tn(LLM_TENSOR_ATTN_V,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wo_in_s && layer.wo) {
                layer.wo_in_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_in_s && layer.wqkv) {
                layer.wqkv_in_s = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_gate_in_s && layer.wqkv_gate) {
                layer.wqkv_gate_in_s = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_in_s && layer.ffn_gate) {
                layer.ffn_gate_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_in_s && layer.ffn_down) {
                layer.ffn_down_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_in_s && layer.ffn_up) {
                layer.ffn_up_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_exps_in_s && layer.ffn_gate_exps) {
                layer.ffn_gate_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_exps_in_s && layer.ffn_down_exps) {
                layer.ffn_down_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_exps_in_s && layer.ffn_up_exps) {
                layer.ffn_up_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_shexp_in_s && layer.ffn_gate_shexp) {
                layer.ffn_gate_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_shexp_in_s && layer.ffn_down_shexp) {
                layer.ffn_down_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_shexp_in_s && layer.ffn_up_shexp) {
                layer.ffn_up_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_in_in_s && layer.ssm_in) {
                layer.ssm_in_in_s = create_tensor(tn(LLM_TENSOR_SSM_IN, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_out_in_s && layer.ssm_out) {
                layer.ssm_out_in_s = create_tensor(tn(LLM_TENSOR_SSM_OUT, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_alpha_in_s && layer.ssm_alpha) {
                layer.ssm_alpha_in_s = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_beta_in_s && layer.ssm_beta) {
                layer.ssm_beta_in_s = create_tensor(tn(LLM_TENSOR_SSM_BETA, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.eh_proj_in_s && layer.nextn.eh_proj) {
                layer.nextn.eh_proj_in_s = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.shared_head_head_in_s && layer.nextn.shared_head_head) {
                layer.nextn.shared_head_head_in_s = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
        }
        // output scales
        if (output && output->type == GGML_TYPE_NVFP4) {
            // weight scale
            if (!output_s) {
                output_s = create_tensor(tn(LLM_TENSOR_OUTPUT, "scale"), {1}, TENSOR_NOT_REQUIRED);
            }
            // input scale
            if (!output_in_s) {
                output_in_s = create_tensor(tn(LLM_TENSOR_OUTPUT, "input_scale"), {1}, TENSOR_NOT_REQUIRED);
            }
        }
    }
    ml.done_getting_tensors();

    // Tied NVFP4 output is valid when no separate LM-head scale tensors are present.
    // If sidecar scales exist, the output weight must be an actual output tensor.
    GGML_ASSERT(!(output && tok_embd &&
            strcmp(output->name, tok_embd->name) == 0 &&
            output->type == GGML_TYPE_NVFP4 &&
            (output_s || output_in_s)));
    // populate tensors_by_name
    for (auto & [_, ctx_ptr] : ml.ctx_map) {
        for (auto * cur = ggml_get_first_tensor(ctx_ptr.get()); cur != NULL; cur = ggml_get_next_tensor(ctx_ptr.get(), cur)) {
            tensors_by_name.emplace_back(ggml_get_name(cur), cur);
        }
    }

    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);
    pimpl->mappings.reserve(ml.mappings.size());

    // create the backend buffers
    std::vector<std::pair<ggml_context *, llama_buf_map>> ctx_buf_maps;
    ctx_buf_maps.reserve(ml.ctx_map.size());

    // Ensure we have enough capacity for the maximum backend buffer we will potentially create
    const size_t n_max_backend_buffer = ml.ctx_map.size() * ml.files.size();
    pimpl->ctxs_bufs.reserve(n_max_backend_buffer);

    for (auto & [ctx_key, ctx_ptr] : ml.ctx_map) {
        ggml_backend_buffer_type_t buft = ctx_key.buft;
        ggml_context * ctx = ctx_ptr.get();

        // skip contexts without tensors
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        llama_buf_map buf_map;
        buf_map.reserve(n_max_backend_buffer);

        // check if it is possible to use buffer_from_host_ptr with this buffer type
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            // FIXME: workaround for CPU backend buft having a NULL device
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!dev) {
                throw std::runtime_error(format("%s: no CPU backend found", __func__));
            }
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
        bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);

        std::vector<ggml_backend_buffer_ptr> bufs;

        // a lazy context is mapped whatever the load mode, but the memory-fit pass maps nothing
        const bool is_lazy_mapped = ctx_key.lazy && !ml.no_alloc;

        ggml_backend_reg_t buft_reg = ggml_backend_dev_backend_reg(dev);
        auto is_moe_cache_buft_fn = buft_reg != nullptr ?
            (ggml_backend_moe_cache_is_buffer_type_t) ggml_backend_reg_get_proc_address(
                buft_reg, GGML_BACKEND_MOE_CACHE_IS_BUFFER_TYPE_PROC_NAME) : nullptr;
        const bool is_moe_cache_buft = is_moe_cache_buft_fn != nullptr && is_moe_cache_buft_fn(buft);
        if (!ml.no_alloc && ml.use_mmap && use_mmap_buffer && is_moe_cache_buft) {
            auto buffer_from_host_ptr_fn = (ggml_backend_moe_cache_buffer_from_host_ptr_t)
                    ggml_backend_reg_get_proc_address(buft_reg, GGML_BACKEND_MOE_CACHE_BUFFER_FROM_HOST_PTR_PROC_NAME);
            if (buffer_from_host_ptr_fn == nullptr) {
                throw std::runtime_error(format("%s does not support buffers from mapped host memory", ggml_backend_buft_name(buft)));
            }
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                void * addr = nullptr;
                size_t first, last; // NOLINT
                ml.get_mapping_range(&first, &last, &addr, idx, ctx);
                if (first >= last) {
                    continue;
                }
                ggml_backend_buffer_t buf = buffer_from_host_ptr_fn(buft, (char *) addr + first, last - first);
                if (buf == nullptr) {
                    throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
                }
                bufs.emplace_back(buf);
                buf_map.emplace(idx, buf);
            }
        } else if (!ml.no_alloc && (ml.use_mmap || is_lazy_mapped) && use_mmap_buffer &&
                   buffer_from_host_ptr_supported && is_default_buft) {
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                // only the mmap region containing the tensors in the model is mapped to the backend buffer
                // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer,
                //     then we could just use metal for all layers
                // this allows using partial offloading when the model size exceeds the metal buffer size, but not the RAM size
                void * addr = nullptr;
                size_t first, last; // NOLINT
                ml.get_mapping_range(&first, &last, &addr, idx, ctx);
                if (first >= last) {
                    continue;
                }
                const size_t max_size = ggml_get_max_tensor_size(ctx);
                ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, (char *) addr + first, last - first, max_size);
                if (buf == nullptr) {
                    throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
                }
                bufs.emplace_back(buf);
                buf_map.emplace(idx, buf);
            }
        } else {
            ggml_backend_buffer_t buf;
            if (ml.no_alloc) {
                buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    t->buffer = buf; // set dummy buffer for weights so that the backend scheduler won't try to allocate them
                }
            } else {
                buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft); // real buffer
            }
            if (buf == nullptr) {
                throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
            }
            if (use_mlock && ggml_backend_buffer_is_host(buf)) {
                pimpl->mlock_bufs.emplace_back(new llama_mlock);
                auto & mlock_buf = pimpl->mlock_bufs.back();
                mlock_buf->init   (ggml_backend_buffer_get_base(buf));
                mlock_buf->grow_to(ggml_backend_buffer_get_size(buf));
            }
            bufs.emplace_back(buf);
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                buf_map.emplace(idx, buf);
            }
        }

        for (auto & buf : bufs) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
            ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        pimpl->ctxs_bufs.emplace_back(std::move(ctx_ptr), std::move(bufs));

        ctx_buf_maps.emplace_back(ctx, buf_map);
    }

    if (llama_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, n_layer_all);

        int n_repeating = n_gpu;
        if (n_repeating > 0) {
            LLAMA_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
            n_repeating--;
        }
        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_repeating);

        const int max_backend_supported_layers = n_layer_all + 1;
        const int max_offloadable_layers       = n_layer_all + 1;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
    }

    // print memory requirements per buffer type
    for (auto & [_, bufs] : pimpl->ctxs_bufs) {
        for (auto & buf: bufs) {
            LLAMA_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
        }
    }

    build_moe_sources();
    finalize_moe_expert_cache();

    if (ml.no_alloc) {
        return true;
    }

    // without mmap, load non-host buffers first: their tensors go through a staging buffer, which is cheapest while the fewest weights are resident
    if (!ml.use_mmap) {
        std::stable_partition(ctx_buf_maps.begin(), ctx_buf_maps.end(), [](const auto & ctx_buf_map) {
            const auto & buf_map = ctx_buf_map.second;
            return !buf_map.empty() && !ggml_backend_buffer_is_host(buf_map.begin()->second);
        });
    }

    // load tensor data
    for (auto & [ctx, buf_map] : ctx_buf_maps) {
        if (!ml.load_all_data(ctx, buf_map, use_mlock ? &pimpl->mlock_mmaps : NULL, params.progress_callback, params.progress_callback_user_data)) {
            return false;
        }
    }

    if (params.moe_expert_cache_host_pinned_size > 0) {
        std::vector<ggml_backend_moe_candidate_group_v2> groups;
        std::vector<ggml_backend_moe_candidate_tensor_v2> tensors;
        std::unordered_set<ggml_backend_buffer_type_t> owners;
        std::unordered_set<const ggml_tensor *> seen;
        for (const auto & group : moe_sources()) {
            const uint32_t index = groups.size();
            groups.push_back({group.layout, group.domain, group.route_present ? 0u : uint32_t(GGML_BACKEND_MOE_CANDIDATE_GROUP_V2_FLAG_INCOMPLETE), 0});
            for (const auto & bank : group.banks) {
                tensors.push_back({bank.tensor, index, bank.role, bank.status, 0, 0});
                seen.insert(bank.tensor);
            }
        }
        for (const auto & named : tensors_by_name) {
            auto * tensor = named.second;
            if (tensor->buffer == nullptr) {
                continue;
            }
            auto buft = ggml_backend_buffer_get_type(tensor->buffer);
            auto dev = ggml_backend_buft_get_device(buft);
            auto reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
            auto is_cached = reg != nullptr ? (ggml_backend_moe_cache_is_buffer_type_t) ggml_backend_reg_get_proc_address(
                reg, GGML_BACKEND_MOE_CACHE_IS_BUFFER_TYPE_PROC_NAME) : nullptr;
            if (is_cached != nullptr && is_cached(buft)) {
                owners.insert(buft);
                if (seen.insert(tensor).second) {
                    tensors.push_back({tensor, UINT32_MAX, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_INVALID, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_UNCLASSIFIED, 0, 0});
                }
            }
        }
        ggml_backend_moe_candidate_snapshot_v2 sources = {};
        sources.magic = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_MAGIC;
        sources.abi_version = GGML_BACKEND_MOE_CANDIDATE_SNAPSHOT_V2_VERSION;
        sources.struct_size = sizeof(sources);
        sources.groups = groups.data();
        sources.n_groups = groups.size();
        sources.tensors = tensors.data();
        sources.n_tensors = tensors.size();
        for (auto buft : owners) {
            sources.n_slots = 0;
            for (const auto & entry : pimpl->moe_cache_slots) {
                sources.n_slots = std::max(sources.n_slots, static_cast<uint32_t>(entry.second));
            }
            auto reg = ggml_backend_dev_backend_reg(ggml_backend_buft_get_device(buft));
            auto configure = (ggml_backend_moe_cache_configure_sources_t) ggml_backend_reg_get_proc_address(
                reg, GGML_BACKEND_MOE_CACHE_CONFIGURE_SOURCES_PROC_NAME);
            if (configure == nullptr || !configure(buft, &sources)) {
                throw std::runtime_error("unable to reserve the bounded MoE host source budget");
            }
        }
    }

    if (use_mmap_buffer) {
        for (auto & mapping : ml.mappings) {
            pimpl->mappings.emplace_back(std::move(mapping));
        }
    }

    return true;
}

ggml_tensor * llama_model_base::create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;
    ggml_tensor * tensor = ml.create_tensor(
        hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,
        tn, ne, flags);
    if (tensor != nullptr && tn.bid >= 0) {
        const auto resolution = ml.tensor_override_resolutions.find(tn.str());
        if (resolution != ml.tensor_override_resolutions.end() &&
                (resolution->second.origin == llama_model_loader::TENSOR_OVERRIDE_CACHE_LEGACY ||
                 resolution->second.origin == llama_model_loader::TENSOR_OVERRIDE_CACHE_SELECTOR) &&
                resolution->second.selected_buft == resolution->second.resolved_buft) {
            // The shared host buffer identifies its provider, not the layer's execution device.
            ggml_backend_dev_t owner = pimpl->dev_layer.at(tn.bid).dev;
            if (ggml_backend_dev_type(owner) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                owner = ggml_backend_buft_get_device(ml.tensor_buft_overrides[resolution->second.index].buft);
            }
            pimpl->moe_cache_tensor_owners.emplace(tensor, owner);
        }
    }
    return tensor;
}

bool llama_model::graph_supports_recurrent_sparse_snapshots() const {
    return false;
}

void llama_model::prefetch_rows(const ggml_tensor * tensor, const int32_t * rows, size_t n_rows) const {
    if (!tensor || !tensor->data || !ggml_is_matrix(tensor) || !ggml_is_contiguous(tensor)) {
        return;
    }
    for (const auto & mapping : pimpl->mappings) {
        mapping->prefetch_rows(tensor->data, tensor->nb[1], rows, n_rows);
    }
}

void llama_model::prefetch_rows(const ggml_tensor * tensor, const ggml_tensor * indices) const {
    for (const auto & mapping : pimpl->mappings) {
        mapping->prefetch_rows(tensor, indices);
    }
}

std::string llama_model::arch_name() const {
    return llm_arch_name(arch);
}

std::string llama_model::type_name() const {
    return llm_type_name(type);
}

std::string llama_model::desc() const {
    return pimpl->desc_str;
}

llama_ftype llama_model::ftype() const {
    return pimpl->ftype;
}

size_t llama_model::size() const {
    return pimpl->n_bytes;
}

size_t llama_model::n_tensors() const {
    return tensors_by_name.size();
}

size_t llama_model::n_devices() const {
    return devices.size();
}

const float * llama_model::tensor_split() const {
    return params.tensor_split;
}

uint32_t llama_model::n_gpu_layers() const {
    // note: plus 1 for the "output" layer
    return params.n_gpu_layers >= 0 ? params.n_gpu_layers : hparams.n_layer_all + 1;
}

llama_split_mode llama_model::split_mode() const {
    return params.split_mode;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_model::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, bufs] : pimpl->ctxs_bufs) {
        if (hparams.no_alloc) {
            GGML_ASSERT(bufs.size() == 1);
            ggml_backend_buffer_t buf = bufs[0].get();
            GGML_ASSERT(ggml_backend_buffer_get_base(buf) == nullptr);
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
            const auto accounting_buft = is_moe_cache_source_buft(buft) ? ggml_backend_cpu_buffer_type() : buft;
            ret[accounting_buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            for (const auto & buf : bufs) {
                // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
                ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());
                const auto accounting_buft = is_moe_cache_source_buft(buft) ? ggml_backend_cpu_buffer_type() : buft;
                ret[accounting_buft] += ggml_backend_buffer_get_size(buf.get());
            }
        }
    }
    return ret;
}

uint64_t llama_model::n_elements() const {
    return pimpl->n_elements;
}

void llama_model::print_info() const {
    const std::string rope_scaling_type = llama_rope_scaling_type_name(hparams.rope_scaling_type_train);

    auto print_f = [](const std::function<int32_t(uint32_t)> & f, uint32_t n) {
        bool is_var = false;

        std::vector<int32_t> v;
        for (uint32_t i = 0; i < n; ++i) {
            v.push_back(f(i));
            if (v[i] != v[0]) {
                is_var = true;
            }
        }

        std::stringstream ss;

        if (is_var) {
            ss << "[";
            for (uint32_t i = 0; i < n; ++i) {
                ss << v[i];
                if (i < n - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
        } else {
            ss << v[0];
        }

        return ss.str();
    };

    // hparams
    LLAMA_LOG_INFO("%s: arch                  = %s\n",     __func__, arch_name().c_str());
    LLAMA_LOG_INFO("%s: vocab_only            = %d\n",     __func__, hparams.vocab_only);
    LLAMA_LOG_INFO("%s: no_alloc              = %d\n",     __func__, hparams.no_alloc);

    if (!hparams.vocab_only) {
        LLAMA_LOG_INFO("%s: n_ctx_train           = %u\n",     __func__, hparams.n_ctx_train);
        LLAMA_LOG_INFO("%s: n_embd_inp            = %u\n",     __func__, hparams.n_embd_inp());
        LLAMA_LOG_INFO("%s: n_embd                = %u\n",     __func__, hparams.n_embd);
        LLAMA_LOG_INFO("%s: n_embd_out            = %u\n",     __func__, hparams.n_embd_out());
        LLAMA_LOG_INFO("%s: n_layer               = %u\n",     __func__, hparams.n_layer());
        LLAMA_LOG_INFO("%s: n_layer_all           = %u\n",     __func__, hparams.n_layer_all);
        LLAMA_LOG_INFO("%s: n_head                = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head(il);    }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_head_kv             = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head_kv(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_rot                 = %u\n",     __func__, hparams.n_rot_full);
        LLAMA_LOG_INFO("%s: n_swa                 = %u\n",     __func__, hparams.n_swa);
        LLAMA_LOG_INFO("%s: is_swa_any            = %u\n",     __func__, hparams.is_swa_any());
        LLAMA_LOG_INFO("%s: non_causal_type       = %d\n",     __func__, hparams.non_causal_type);
        LLAMA_LOG_INFO("%s: n_embd_head_k         = %u\n",     __func__, hparams.n_embd_head_k_full);
        LLAMA_LOG_INFO("%s: n_embd_head_v         = %u\n",     __func__, hparams.n_embd_head_v_full);
        LLAMA_LOG_INFO("%s: n_gqa                 = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_gqa(il);        }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_embd_k_gqa          = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_k_gqa(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_embd_v_gqa          = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_v_gqa(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: f_norm_eps            = %.1e\n",   __func__, hparams.f_norm_eps);
        LLAMA_LOG_INFO("%s: f_norm_rms_eps        = %.1e\n",   __func__, hparams.f_norm_rms_eps);
        LLAMA_LOG_INFO("%s: f_clamp_kqv           = %.1e\n",   __func__, hparams.f_clamp_kqv);
        LLAMA_LOG_INFO("%s: f_max_alibi_bias      = %.1e\n",   __func__, hparams.f_max_alibi_bias);
        LLAMA_LOG_INFO("%s: f_logit_scale         = %.1e\n",   __func__, hparams.f_logit_scale);
        LLAMA_LOG_INFO("%s: f_attn_scale          = %.1e\n",   __func__, hparams.f_attention_scale);
        LLAMA_LOG_INFO("%s: f_attn_value_scale    = %.4f\n",   __func__, hparams.f_attn_value_scale);
        LLAMA_LOG_INFO("%s: n_ff                  = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_ff(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_expert              = %u\n",     __func__, hparams.n_expert);
        LLAMA_LOG_INFO("%s: n_expert_used         = %u\n",     __func__, hparams.n_expert_used());
        LLAMA_LOG_INFO("%s: n_expert_groups       = %d\n",     __func__, hparams.n_expert_groups);
        LLAMA_LOG_INFO("%s: n_group_used          = %d\n",     __func__, hparams.n_group_used);
        LLAMA_LOG_INFO("%s: causal attn           = %d\n",     __func__, hparams.causal_attn);
        LLAMA_LOG_INFO("%s: pooling type          = %d\n",     __func__, hparams.pooling_type);
        LLAMA_LOG_INFO("%s: rope type             = %d\n",     __func__, hparams.rope_type);
        LLAMA_LOG_INFO("%s: rope scaling          = %s\n",     __func__, rope_scaling_type.c_str());
        LLAMA_LOG_INFO("%s: freq_base_train       = %.1f\n",   __func__, hparams.rope_freq_base_train);
        LLAMA_LOG_INFO("%s: freq_scale_train      = %g\n",     __func__, hparams.rope_freq_scale_train);
        if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
            LLAMA_LOG_INFO("%s: freq_base_swa         = %.1f\n",   __func__, hparams.rope_freq_base_train_swa);
            LLAMA_LOG_INFO("%s: freq_scale_swa        = %g\n",     __func__, hparams.rope_freq_scale_train_swa);
            LLAMA_LOG_INFO("%s: n_embd_head_k_swa     = %u\n",     __func__, hparams.n_embd_head_k_swa);
            LLAMA_LOG_INFO("%s: n_embd_head_v_swa     = %u\n",     __func__, hparams.n_embd_head_v_swa);
            LLAMA_LOG_INFO("%s: n_rot_swa             = %u\n",     __func__, hparams.n_rot_swa);
        }
        LLAMA_LOG_INFO("%s: n_ctx_orig_yarn       = %u\n",     __func__, hparams.n_ctx_orig_yarn);
        LLAMA_LOG_INFO("%s: rope_yarn_log_mul     = %.4f\n",   __func__, hparams.rope_yarn_log_mul);
        LLAMA_LOG_INFO("%s: rope_finetuned        = %s\n",     __func__, hparams.rope_finetuned ? "yes" : "unknown");
        if (arch == LLM_ARCH_GRANITE &&
            std::any_of(hparams.deepstack_mapping_arr.begin(),
                        hparams.deepstack_mapping_arr.end(),
                        [](const auto & entry) { return entry >= 0; })) {
            LLAMA_LOG_INFO("%s: deepstack_mapping_arr = %s\n", __func__,
                           print_f([&](uint32_t il) { return hparams.deepstack_mapping_arr[il]; },
                           hparams.n_layer_all).c_str());
        }
        // MRoPE (Multi-axis Rotary Position Embedding) sections
        if (const auto & s = hparams.rope_sections; s[0] || s[1] || s[2] || s[3]) {
            LLAMA_LOG_INFO("%s: mrope sections        = [%d, %d, %d, %d]\n", __func__, s[0], s[1], s[2], s[3]);
        }
        if (!classifier_labels.empty()) {
            LLAMA_LOG_INFO("%s: n_cls_out             = %u\n", __func__, hparams.n_cls_out);

            size_t i = 0;
            for (const auto & label : classifier_labels) {
                LLAMA_LOG_INFO("%s: cls_label[%2zu]         = %s\n", __func__, i++, label.c_str());
            }
        }

        if (arch == LLM_ARCH_MAMBA ||
                arch == LLM_ARCH_MAMBA2 ||
                arch == LLM_ARCH_JAMBA ||
                arch == LLM_ARCH_FALCON_H1 ||
                arch == LLM_ARCH_PLAMO2 ||
                arch == LLM_ARCH_GRANITE_HYBRID ||
                arch == LLM_ARCH_QWEN3NEXT ||
                arch == LLM_ARCH_QWEN35 ||
                arch == LLM_ARCH_QWEN35MOE ||
                arch == LLM_ARCH_NEMOTRON_H ||
                arch == LLM_ARCH_NEMOTRON_H_MOE) {
            LLAMA_LOG_INFO("%s: ssm_d_conv            = %u\n",     __func__, hparams.ssm_d_conv);
            LLAMA_LOG_INFO("%s: ssm_d_inner           = %u\n",     __func__, hparams.ssm_d_inner);
            LLAMA_LOG_INFO("%s: ssm_d_state           = %u\n",     __func__, hparams.ssm_d_state);
            LLAMA_LOG_INFO("%s: ssm_dt_rank           = %u\n",     __func__, hparams.ssm_dt_rank);
            LLAMA_LOG_INFO("%s: ssm_n_group           = %u\n",     __func__, hparams.ssm_n_group);
            LLAMA_LOG_INFO("%s: ssm_dt_b_c_rms        = %d\n",     __func__, hparams.ssm_dt_b_c_rms);
        }

        LLAMA_LOG_INFO("%s: model type            = %s\n",     __func__, type_name().c_str());
        if (pimpl->n_elements >= 1e12) {
            LLAMA_LOG_INFO("%s: model params          = %.2f T\n", __func__, pimpl->n_elements*1e-12);
        } else if (pimpl->n_elements >= 1e9) {
            LLAMA_LOG_INFO("%s: model params          = %.2f B\n", __func__, pimpl->n_elements*1e-9);
        } else if (pimpl->n_elements >= 1e6) {
            LLAMA_LOG_INFO("%s: model params          = %.2f M\n", __func__, pimpl->n_elements*1e-6);
        } else {
            LLAMA_LOG_INFO("%s: model params          = %.2f K\n", __func__, pimpl->n_elements*1e-3);
        }

        // general kv
        LLAMA_LOG_INFO("%s: general.name          = %s\n",    __func__, name.c_str());

        if (arch == LLM_ARCH_DEEPSEEK) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
        }

        if (arch == LLM_ARCH_DEEPSEEK2 || arch == LLM_ARCH_DEEPSEEK2OCR ||
                arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA ||
                arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_MISTRAL4 ||
                arch == LLM_ARCH_HY_V4) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_lora_q              = %d\n",     __func__, hparams.n_lora_q);
            LLAMA_LOG_INFO("%s: n_lora_kv             = %d\n",     __func__, hparams.n_lora_kv);
            LLAMA_LOG_INFO("%s: n_embd_head_k_mla     = %d\n",     __func__, hparams.n_embd_head_k_mla());
            LLAMA_LOG_INFO("%s: n_embd_head_v_mla     = %d\n",     __func__, hparams.n_embd_head_v_mla());
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
            LLAMA_LOG_INFO("%s: expert_weights_norm   = %d\n",     __func__, hparams.expert_weights_norm);
            LLAMA_LOG_INFO("%s: expert_gating_func    = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
        }

        if (arch == LLM_ARCH_QWEN2MOE) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_ff_shexp            = %d\n",     __func__, hparams.n_ff_shexp);
        }

        if (arch == LLM_ARCH_MELLUM ||
                arch == LLM_ARCH_COHERE2MOE ||
                arch == LLM_ARCH_QWEN3MOE ||
                arch == LLM_ARCH_OPENAI_MOE ||
                arch == LLM_ARCH_QWEN3VLMOE ||
                arch == LLM_ARCH_RND1) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
        }

        if (arch == LLM_ARCH_MINICPM ||
                arch == LLM_ARCH_GRANITE ||
                arch == LLM_ARCH_GRANITE_MOE ||
                arch == LLM_ARCH_GRANITE_HYBRID ||
                arch == LLM_ARCH_GRANITE_SWITCH ||
                arch == LLM_ARCH_NEMOTRON_H_MOE) {
            LLAMA_LOG_INFO("%s: f_embedding_scale     = %f\n", __func__, hparams.f_embedding_scale);
            LLAMA_LOG_INFO("%s: f_residual_scale      = %f\n", __func__, hparams.f_residual_scale);
            LLAMA_LOG_INFO("%s: f_attention_scale     = %f\n", __func__, hparams.f_attention_scale);
            LLAMA_LOG_INFO("%s: n_ff_shexp            = %d\n", __func__, hparams.n_ff_shexp);
        }

        if (arch == LLM_ARCH_BAILINGMOE) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
            LLAMA_LOG_INFO("%s: expert_weights_norm   = %d\n",     __func__, hparams.expert_weights_norm);
        }

        if (arch == LLM_ARCH_BAILINGMOE2 || arch == LLM_ARCH_BAILINGMOE3) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_ff_shexp            = %d\n",     __func__, hparams.n_ff_shexp);
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
            LLAMA_LOG_INFO("%s: expert_weights_norm   = %d\n",     __func__, hparams.expert_weights_norm);
            LLAMA_LOG_INFO("%s: expert_gating_func    = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
            LLAMA_LOG_INFO("%s: n_layer_nextn         = %d\n",     __func__, hparams.n_layer_nextn);
        }

        if (arch == LLM_ARCH_SMALLTHINKER || arch == LLM_ARCH_LFM2MOE) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: expert_gating_func    = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
        }

        if (arch == LLM_ARCH_GROVEMOE) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_ff_chexp            = %d\n",     __func__, hparams.n_ff_chexp);
            LLAMA_LOG_INFO("%s: n_group_experts       = %d\n",     __func__, hparams.n_group_experts);
            LLAMA_LOG_INFO("%s: expert_group_scale    = %.2f\n",   __func__, hparams.expert_group_scale);
        }
    }

    vocab.print_info();
}

ggml_backend_dev_t llama_model::dev_layer(int il) const {
    return pimpl->dev_layer.at(il).dev;
}

ggml_backend_dev_t llama_model::dev_output() const {
    return pimpl->dev_output.dev;
}

template<typename F>
static bool buft_supported(ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev, F & fn) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        throw std::runtime_error(format("failed to create ggml context"));
    }

    ggml_backend_buffer_ptr buf { ggml_backend_buft_alloc_buffer(buft, 0) };
    ggml_tensor * op_tensor = fn(ctx.get());
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op_tensor->src[i] != nullptr) {
            assert(op_tensor->src[i]->buffer == nullptr);
            op_tensor->src[i]->buffer = buf.get();
        }
    }

    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);

    return op_supported;
}

template<typename F>
static ggml_backend_buffer_type_t select_buft(const buft_list_t & buft_list, const F & fn) {
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (buft_supported(cur_buft, cur_dev, fn)) {
            return cur_buft;
        }
    }

    throw std::runtime_error(format("no suitable buffer type found"));
}

ggml_backend_buffer_type_t llama_model::select_buft(int il) const {
    return ::select_buft(
            *pimpl->dev_layer.at(il).buft_list,
            [&](ggml_context * ctx) {
                ggml_tensor * cur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                ggml_tensor * layer_dir = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                return ggml_add(ctx, cur, layer_dir);
            });
}

bool llama_model::has_tensor_overrides() const {
    return pimpl->has_tensor_overrides;
}

bool llama_model::is_no_alloc() const {
    return params.no_alloc;
}

void llama_model::record_tensor_override_resolution(
        std::string tensor_name, uint32_t origin, int32_t index, std::string pattern,
        std::string requested_buft, std::string selected_buft, std::string resolved_buft,
        std::string resolved_owner_canonical_id, std::string resolved_owner_identity_kind,
        std::string resolved_owner_backend, bool resolved_is_host) {
    impl::tensor_override_resolution resolution;
    resolution.origin = origin;
    resolution.index = index;
    resolution.pattern = std::move(pattern);
    resolution.requested_buft = std::move(requested_buft);
    resolution.selected_buft = std::move(selected_buft);
    resolution.resolved_buft = std::move(resolved_buft);
    resolution.resolved_owner_canonical_id = std::move(resolved_owner_canonical_id);
    resolution.resolved_owner_identity_kind = std::move(resolved_owner_identity_kind);
    resolution.resolved_owner_backend = std::move(resolved_owner_backend);
    resolution.resolved_is_host = resolved_is_host;
    pimpl->tensor_override_resolutions.emplace(std::move(tensor_name), std::move(resolution));
}

void llama_model::record_shared_tensor(
        std::string tensor_name, size_t tensor_bytes, int32_t type,
        std::array<int64_t, GGML_MAX_DIMS> ne, std::array<size_t, GGML_MAX_DIMS> nb,
        std::string resolved_buft, std::string resolved_class,
        std::string owner_canonical_id, std::string owner_identity_kind,
        std::string owner_backend, bool resolved_storage_available,
        bool current_storage_available, std::string storage_provenance) {
    llama_moe_shared_tensor shared;
    shared.name = tensor_name;
    shared.tensor_bytes = tensor_bytes;
    shared.type = type;
    shared.ne = ne;
    shared.nb = nb;
    shared.resolved_buft = std::move(resolved_buft);
    shared.resolved_class = std::move(resolved_class);
    shared.owner_canonical_id = std::move(owner_canonical_id);
    shared.owner_identity_kind = std::move(owner_identity_kind);
    shared.owner_backend = std::move(owner_backend);
    shared.storage_relation = "borrowed_model_shared";
    shared.resolved_storage_available = resolved_storage_available;
    shared.current_storage_available = current_storage_available;
    shared.storage_provenance = std::move(storage_provenance);
    pimpl->borrowed_tensors.emplace(std::move(tensor_name), std::move(shared));
}

void llama_model::record_artifact_source(
        size_t file_size, int64_t modification_time, bool modification_time_available) {
    pimpl->artifact_sources.push_back({file_size, modification_time, modification_time_available});
}

void llama_model::record_resolved_load_strategy(
        bool uses_mmap, std::string direct_io_state, bool uses_mlock, bool has_lazy_tensors) {
    pimpl->resolved_uses_mmap = uses_mmap;
    pimpl->resolved_direct_io_state = std::move(direct_io_state);
    pimpl->resolved_uses_mlock = uses_mlock;
    pimpl->resolved_has_lazy_tensors = has_lazy_tensors;
}

int32_t llama_model::moe_expert_cache_slots() const {
    if (!pimpl->moe_cache_byte_budgets_owned.empty() && !pimpl->moe_cache_slots.empty()) {
        int32_t result = INT32_MAX;
        for (const auto & entry : pimpl->moe_cache_slots) {
            result = std::min(result, entry.second);
        }
        return result;
    }
    return params.moe_expert_cache_slots;
}

size_t llama_moe_cache_memory::device_bytes(uint32_t slots) const {
    if (per_slot_device_bytes != 0 && slots > (SIZE_MAX - fixed_device_bytes) / per_slot_device_bytes) {
        throw std::overflow_error("MoE cache memory estimate overflow");
    }
    return fixed_device_bytes + slots * per_slot_device_bytes;
}

int32_t llama_model::moe_expert_cache_slots(ggml_backend_dev_t dev) const {
    // Preserve the legacy global slot-count contract exactly. Owner-local counts
    // are only meaningful when byte-budget mode selected them independently.
    if (pimpl->moe_cache_byte_budgets_owned.empty()) {
        return params.moe_expert_cache_slots;
    }
    const auto found = pimpl->moe_cache_slots.find(dev);
    return found != pimpl->moe_cache_slots.end() ? found->second : 0;
}

bool llama_model::moe_expert_cache_enabled() const {
    return params.moe_expert_cache_slots > 0;
}

const std::map<ggml_backend_dev_t, llama_moe_cache_memory> & llama_model::moe_expert_cache_memory() const {
    return pimpl->moe_cache_memory;
}

std::map<ggml_backend_dev_t, size_t> llama_model::moe_expert_cache_host_staging(
        enum llama_context_type ctx_type) const {
    std::map<ggml_backend_dev_t, size_t> result;
    if (params.moe_expert_cache_host_pinned_size == 0) {
        return result;
    }
    const size_t context_index = ctx_type == LLAMA_CONTEXT_TYPE_MTP ? 1 : 0;
    std::map<ggml_backend_dev_t, uint32_t> prediction_capacity;
    for (const auto & group : pimpl->moe_cache_staging_groups) {
        if ((group.context_mask & (uint32_t{1} << context_index)) != 0) {
            prediction_capacity[group.owner] = std::max(prediction_capacity[group.owner], group.n_experts);
        }
    }
    for (const auto & group : pimpl->moe_cache_staging_groups) {
        if ((group.context_mask & (uint32_t{1} << context_index)) == 0) {
            continue;
        }
        auto * reg = ggml_backend_dev_backend_reg(group.owner);
        auto staging_size = reg != nullptr ? reinterpret_cast<ggml_backend_moe_staging_size_v1_t>(
            ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_STAGING_SIZE_V1_PROC_NAME)) : nullptr;
        const int32_t slots = moe_expert_cache_slots(group.owner);
        if (staging_size == nullptr || slots <= 0 || group.bank_expert_strides.empty() ||
                group.bank_expert_strides.size() > GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS) {
            continue;
        }
        ggml_backend_moe_staging_query_v1 query = {};
        query.struct_size = sizeof(query);
        query.n_slots = static_cast<uint32_t>(slots);
        query.n_experts = prediction_capacity[group.owner];
        query.top_k = group.top_k;
        query.n_banks = static_cast<uint32_t>(group.bank_expert_strides.size());
        query.staged_bank_mask = (uint32_t{1} << query.n_banks) - 1;
        query.family_mask = GGML_BACKEND_MOE_STAGING_FAMILY_V1_GROUPED |
            GGML_BACKEND_MOE_STAGING_FAMILY_V1_LEGACY |
            GGML_BACKEND_MOE_STAGING_FAMILY_V1_HOST_STAGED;
        // The control block can be enabled after model load, so include it in
        // accounting regardless of the process-global runtime setting.
        query.flags = GGML_BACKEND_MOE_STAGING_FLAG_V1_PREDICTION_CONTROL;
        std::copy(group.bank_expert_strides.begin(), group.bank_expert_strides.end(), query.bank_expert_strides);
        ggml_backend_moe_staging_size_v1 sizing = {};
        sizing.struct_size = sizeof(sizing);
        if (!staging_size(&query, &sizing)) {
            throw std::runtime_error("failed to size MoE cache host staging");
        }
        uint64_t staging = sizing.grouped_min_bytes;
        if (sizing.legacy_min_bytes > UINT64_MAX - staging) {
            throw std::overflow_error("MoE cache host staging estimate overflow");
        }
        staging += sizing.legacy_min_bytes;
        if (sizing.host_staged_min_bytes > UINT64_MAX - staging) {
            throw std::overflow_error("MoE cache host staging estimate overflow");
        }
        staging += sizing.host_staged_min_bytes;
        if (staging > SIZE_MAX || static_cast<size_t>(staging) > SIZE_MAX - result[group.owner]) {
            throw std::overflow_error("MoE cache host staging estimate overflow");
        }
        result[group.owner] += static_cast<size_t>(staging);
    }
    return result;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_model::moe_expert_cache_memory_breakdown(
        enum llama_context_type ctx_type) const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    const size_t context_index = ctx_type == LLAMA_CONTEXT_TYPE_MTP ? 1 : 0;
    size_t host_staging = 0;
    for (const auto & entry : pimpl->moe_cache_context_memory[context_index]) {
        const auto slots = moe_expert_cache_slots(entry.first);
        if (slots > 0) {
            result[ggml_backend_dev_buffer_type(entry.first)] += entry.second.device_bytes(slots);
        }
    }
    for (const auto & [owner, bytes] : moe_expert_cache_host_staging(ctx_type)) {
        (void) owner;
        if (bytes > SIZE_MAX - host_staging) {
            throw std::overflow_error("MoE cache host staging estimate overflow");
        }
        host_staging += bytes;
    }
    if (pimpl->moe_cache_context_host_memory[context_index] > SIZE_MAX - host_staging) {
        throw std::overflow_error("MoE cache host memory estimate overflow");
    }
    host_staging += pimpl->moe_cache_context_host_memory[context_index];
    if (host_staging != 0) {
        result[ggml_backend_cpu_buffer_type()] += host_staging;
    }
    return result;
}

uint32_t llama_moe_placement_context_mask(
        bool load_mtp, uint32_t n_layer_nextn, int32_t router_layer,
        uint32_t n_trunk_layers, int32_t group_layer) {
    const bool mtp_context_active = load_mtp && n_layer_nextn > 0 && router_layer < 0;
    const bool disjoint_mtp_layers = mtp_context_active && n_trunk_layers > 0;
    const bool use_mtp = mtp_context_active &&
        (!disjoint_mtp_layers || static_cast<uint32_t>(group_layer) >= n_trunk_layers);
    const bool use_default = !disjoint_mtp_layers || static_cast<uint32_t>(group_layer) < n_trunk_layers;
    return (use_default ? 1u : 0u) | (use_mtp ? 2u : 0u);
}

void llama_model::finalize_moe_expert_cache() {
    pimpl->moe_cache_memory.clear();
    for (auto & memory : pimpl->moe_cache_context_memory) {
        memory.clear();
    }
    pimpl->moe_cache_context_host_memory.fill(0);
    pimpl->moe_cache_staging_groups.clear();
    pimpl->moe_cache_slots.clear();
    for (auto & memory : pimpl->moe_cache_group_context_memory) {
        memory.assign(pimpl->moe_sources.size(), {});
    }
    if (!moe_expert_cache_enabled()) {
        return;
    }

    const auto add = [](size_t & dst, size_t value) {
        if (value > SIZE_MAX - dst) {
            throw std::overflow_error("MoE cache memory estimate overflow");
        }
        dst += value;
    };
    const auto add64 = [](uint64_t & dst, uint64_t value) {
        if (value > UINT64_MAX - dst) {
            throw std::overflow_error("MoE cache memory estimate overflow");
        }
        dst += value;
    };
    struct context_geometry {
        uint32_t width = 0;
        uint32_t experts = 0;
        uint32_t top_k = 0;
        uint32_t hc_rank = 0;
        uint32_t route_capacity = 0;
        uint32_t row_capacity   = 0;
        uint32_t groups       = 0;
        uint64_t expert_bytes = 0;
    };
    std::array<std::map<ggml_backend_dev_t, context_geometry>, 2> context_geometries;
    std::unordered_set<int32_t> unmatched_selected_layers;
    for (const auto & range : pimpl->moe_cache_layer_ranges_owned) {
        for (int64_t layer = range.first; layer <= range.last; ++layer) {
            unmatched_selected_layers.insert(static_cast<int32_t>(layer));
        }
    }

    for (size_t group_index = 0; group_index < pimpl->moe_sources.size(); ++group_index) {
        const auto & group = pimpl->moe_sources[group_index];
        if (group.layer < 0 || static_cast<size_t>(group.layer) >= pimpl->dev_layer.size()) {
            continue;
        }
        uint32_t n_experts = 0;
        uint32_t base_banks = 0;
        uint32_t cached_base_banks = 0;
        uint64_t group_expert_bytes = 0;
        ggml_backend_dev_t cache_owner = nullptr;
        for (const auto & bank : group.banks) {
            const auto * tensor = bank.tensor;
            if (tensor == nullptr) {
                continue;
            }
            if (bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                ++base_banks;
                ggml_backend_buffer_type_t buft = tensor->buffer != nullptr ?
                    ggml_backend_buffer_get_type(tensor->buffer) : nullptr;
                const auto cached_tensor = pimpl->moe_cache_tensor_owners.find(tensor);
                const bool cached = cached_tensor != pimpl->moe_cache_tensor_owners.end();
                cached_base_banks += cached;
                if (tensor->ne[2] <= 0 || tensor->ne[2] > UINT32_MAX) {
                    throw std::runtime_error("invalid MoE expert count for cache accounting");
                }
                if (n_experts != 0 && n_experts != static_cast<uint32_t>(tensor->ne[2])) {
                    throw std::runtime_error("inconsistent MoE expert count in cache group");
                }
                n_experts = static_cast<uint32_t>(tensor->ne[2]);
                if (cached) {
                    ggml_backend_dev_t owner = cached_tensor->second;
                    if (owner == nullptr || ggml_backend_dev_type(owner) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                        throw std::runtime_error(format("MoE cache tensor %s has no accelerator owner (%s)",
                            tensor->name, buft != nullptr ? ggml_backend_buft_name(buft) : "none"));
                    }
                    if (cache_owner != nullptr && cache_owner != owner) {
                        throw std::runtime_error("MoE cache group spans multiple accelerator owners");
                    }
                    cache_owner = owner;
                }
            }
        }
        if (base_banks != 0) {
            unmatched_selected_layers.erase(group.layer);
        }
        if (cached_base_banks == 0) {
            continue;
        }
        if (cached_base_banks != base_banks || n_experts == 0) {
            throw std::runtime_error("MoE cache placement must select every routed weight in a layer group");
        }
        ggml_backend_dev_t owner = cache_owner;
        if (owner == nullptr) {
            throw std::runtime_error("MoE cache placement has no accelerator owner");
        }
        auto * reg = ggml_backend_dev_backend_reg(owner);
        auto device_size = reg != nullptr ? reinterpret_cast<ggml_backend_moe_device_size_v1_t>(
            ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_DEVICE_SIZE_V1_PROC_NAME)) : nullptr;
        if (device_size == nullptr) {
            throw std::runtime_error("MoE cache backend cannot size device allocations");
        }
        ggml_backend_moe_device_size_query_v1 query = {};
        query.struct_size = sizeof(query);
        query.n_slots = 1;
        query.n_experts = n_experts;
        query.flags = GGML_BACKEND_MOE_DEVICE_SIZE_FLAG_V1_DEBUG;
        std::unordered_set<const ggml_tensor *> slot_auxiliaries;
        std::unordered_set<const ggml_tensor *> original_shadows;
        std::unordered_set<const ggml_tensor *> prefill_copies;
        for (const auto & bank : group.banks) {
            const auto * tensor = bank.tensor;
            if (tensor == nullptr) {
                continue;
            }
            if (bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                if (query.n_banks == GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS || tensor->nb[2] == 0 ||
                        tensor->ne[0] <= 0 || tensor->ne[0] > INT32_MAX) {
                    throw std::runtime_error("invalid MoE routed bank for cache accounting");
                }
                const uint32_t index = query.n_banks++;
                query.bank_expert_strides[index] = tensor->nb[2];
                add64(group_expert_bytes, tensor->nb[2]);
                query.bank_ne0[index] = tensor->ne[0];
                query.bank_types[index] = tensor->type;
                continue;
            }
            if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
                continue;
            }
            if (original_shadows.insert(tensor).second) {
                add64(query.original_shadow_bytes, ggml_nbytes(tensor));
            }
            const bool scalar = ggml_n_dims(tensor) == 1 && tensor->ne[0] == n_experts;
            const bool vector = ggml_n_dims(tensor) == 2 && tensor->ne[1] == n_experts && tensor->ne[0] > 0;
            if ((scalar || vector) && slot_auxiliaries.insert(tensor).second) {
                add64(query.slot_auxiliary_values, scalar ? 1 : static_cast<uint64_t>(tensor->ne[0]));
                query.n_slot_auxiliaries = std::min<uint32_t>(3, query.n_slot_auxiliaries + 1);
            }
            if (bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_BIAS &&
                    prefill_copies.insert(tensor).second) {
                add64(query.prefill_copy_bytes, ggml_nbytes(tensor));
            }
        }
        ggml_backend_moe_device_size_v1 sizing = {};
        sizing.struct_size = sizeof(sizing);
        if (!device_size(&query, &sizing) || sizing.group_fixed_bytes > SIZE_MAX ||
                sizing.group_per_slot_bytes > SIZE_MAX) {
            throw std::runtime_error("failed to size MoE cache device allocations");
        }
        llama_moe_cache_memory group_memory;
        group_memory.max_slots = n_experts;
        group_memory.fixed_device_bytes = static_cast<size_t>(sizing.group_fixed_bytes);
        group_memory.per_slot_device_bytes = static_cast<size_t>(sizing.group_per_slot_bytes);
        const auto merge_memory = [&](llama_moe_cache_memory & memory) {
            memory.max_slots = memory.max_slots == 0 ? group_memory.max_slots :
                std::min(memory.max_slots, group_memory.max_slots);
            add(memory.fixed_device_bytes, group_memory.fixed_device_bytes);
            add(memory.per_slot_device_bytes, group_memory.per_slot_device_bytes);
        };
        const uint32_t context_mask = llama_moe_placement_context_mask(
            params.load_mtp, hparams.n_layer_nextn, hparams.router_layer, hparams.n_layer(), group.layer);
        const bool use_default = (context_mask & 1u) != 0;
        const bool use_mtp = (context_mask & 2u) != 0;
        for (size_t context_index = 0; context_index < 2; ++context_index) {
            if ((context_index == 0 && !use_default) || (context_index == 1 && !use_mtp)) {
                continue;
            }
            merge_memory(pimpl->moe_cache_memory[owner]);
            merge_memory(pimpl->moe_cache_context_memory[context_index][owner]);
            merge_memory(pimpl->moe_cache_group_context_memory[context_index][group_index]);
            auto & geometry = context_geometries[context_index][owner];
            const uint32_t top_k = std::max<uint32_t>(1, hparams.n_expert_used(group.layer));
            if (context_index == 0) {
                const auto * router_input = layers[group.layer].ffn_gate_inp;
                uint64_t width = router_input != nullptr && router_input->ne[0] > 0 ?
                    static_cast<uint64_t>(router_input->ne[0]) : hparams.n_embd;
                if (hparams.dsv4_hc_mult > 1) {
                    width = std::max(width, uint64_t(hparams.n_embd) * hparams.dsv4_hc_mult);
                }
                if (width > UINT32_MAX) {
                    throw std::runtime_error("MoE router width exceeds cache accounting ABI");
                }
                geometry.width = std::max(geometry.width, static_cast<uint32_t>(width));
                geometry.experts = std::max(geometry.experts, n_experts);
                geometry.top_k = std::max(geometry.top_k, top_k);
                geometry.hc_rank = std::max(geometry.hc_rank, hparams.hc_low_rank);
                const uint32_t slot_bound = pimpl->moe_cache_byte_budgets_owned.empty() ?
                    static_cast<uint32_t>(params.moe_expert_cache_slots) : n_experts;
                const uint32_t row_capacity = slot_bound / top_k;
                uint32_t candidate_k = top_k;
                if (const char * value = getenv("GGML_CUDA_MOE_EARLY_ROUTER_CANDIDATE_PERCENT")) {
                    char * end = nullptr;
                    const unsigned long long percent = strtoull(value, &end, 10);
                    if (*value != '\0' && end != nullptr && *end == '\0' && percent != 0 && percent <= UINT32_MAX) {
                        const uint64_t scaled = uint64_t(top_k) * percent;
                        candidate_k = static_cast<uint32_t>(std::min<uint64_t>(n_experts,
                            scaled / 100 + (scaled % 100 != 0)));
                    }
                }
                if (row_capacity != 0) {
                    geometry.row_capacity = std::max(geometry.row_capacity, row_capacity);
                    geometry.route_capacity = std::max(geometry.route_capacity, static_cast<uint32_t>(
                        std::min<uint64_t>(n_experts, uint64_t(row_capacity) * candidate_k)));
                    if (geometry.groups == UINT32_MAX) {
                        throw std::runtime_error("MoE cache early-router group count overflow");
                    }
                    ++geometry.groups;
                }
                geometry.expert_bytes = std::max(geometry.expert_bytes, group_expert_bytes);
            }
        }

        if (params.moe_expert_cache_host_pinned_size > 0) {
            impl::moe_cache_staging_group staging_group = {};
            staging_group.owner = owner;
            staging_group.layer = group.layer;
            staging_group.n_experts = n_experts;
            staging_group.top_k = std::max<uint32_t>(1, hparams.n_expert_used(group.layer));
            staging_group.context_mask = context_mask;
            for (const auto & bank : group.banks) {
                if (bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE && bank.tensor != nullptr) {
                    staging_group.bank_expert_strides.push_back(bank.tensor->nb[2]);
                }
            }
            pimpl->moe_cache_staging_groups.push_back(std::move(staging_group));
        }
    }
    for (size_t context_index = 0; context_index < context_geometries.size(); ++context_index) {
        for (const auto & entry : context_geometries[context_index]) {
            auto * reg = ggml_backend_dev_backend_reg(entry.first);
            auto device_size_v2 =
                reg != nullptr ? reinterpret_cast<ggml_backend_moe_device_size_v2_t>(ggml_backend_reg_get_proc_address(
                                     reg, GGML_BACKEND_MOE_DEVICE_SIZE_V2_PROC_NAME)) :
                                 nullptr;
            auto device_size_v1 =
                reg != nullptr ? reinterpret_cast<ggml_backend_moe_device_size_v1_t>(ggml_backend_reg_get_proc_address(
                                     reg, GGML_BACKEND_MOE_DEVICE_SIZE_V1_PROC_NAME)) :
                                 nullptr;
            if (device_size_v2 == nullptr && device_size_v1 == nullptr) {
                throw std::runtime_error("MoE cache backend cannot size context allocations");
            }
            ggml_backend_moe_device_size_query_v2 query = {};
            query.struct_size = sizeof(query);
            query.flags = GGML_BACKEND_MOE_DEVICE_SIZE_FLAG_V1_DEBUG;
            // Early routing is currently limited to MAIN grouped decode.
            if (context_index == 0 && entry.second.groups != 0) {
                query.early_width = entry.second.width;
                query.early_experts = entry.second.experts;
                query.early_top_k = entry.second.top_k;
                query.early_hc_rank = entry.second.hc_rank;
                query.early_route_capacity = entry.second.route_capacity;
                query.early_row_capacity   = entry.second.row_capacity;
                query.early_groups         = entry.second.groups;
                query.early_expert_bytes   = entry.second.expert_bytes;
            }
            uint64_t context_fixed_bytes = 0;
            uint64_t context_host_bytes  = 0;
            if (device_size_v2 != nullptr) {
                ggml_backend_moe_device_size_v2 sizing = {};
                sizing.struct_size = sizeof(sizing);
                if (!device_size_v2(&query, &sizing)) {
                    throw std::runtime_error("failed to size MoE cache context allocations");
                }
                context_fixed_bytes = sizing.context_fixed_bytes;
                context_host_bytes  = sizing.host_fixed_bytes;
            } else {
                ggml_backend_moe_device_size_query_v1 query_v1 = {};
                query_v1.struct_size = sizeof(query_v1);
                query_v1.n_slots = query.n_slots;
                query_v1.n_experts = query.n_experts;
                query_v1.n_banks = query.n_banks;
                query_v1.n_slot_auxiliaries = query.n_slot_auxiliaries;
                query_v1.flags = query.flags;
                query_v1.early_width = query.early_width;
                query_v1.early_experts = query.early_experts;
                query_v1.early_top_k = query.early_top_k;
                query_v1.early_hc_rank = query.early_hc_rank;
                query_v1.slot_auxiliary_values = query.slot_auxiliary_values;
                query_v1.original_shadow_bytes = query.original_shadow_bytes;
                query_v1.prefill_copy_bytes = query.prefill_copy_bytes;
                std::copy_n(query.bank_expert_strides, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS, query_v1.bank_expert_strides);
                std::copy_n(query.bank_ne0, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS, query_v1.bank_ne0);
                std::copy_n(query.bank_types, GGML_BACKEND_MOE_CANDIDATE_MAX_BANKS, query_v1.bank_types);
                ggml_backend_moe_device_size_v1 sizing = {};
                sizing.struct_size = sizeof(sizing);
                if (!device_size_v1(&query_v1, &sizing)) {
                    throw std::runtime_error("failed to size MoE cache context allocations");
                }
                context_fixed_bytes = sizing.context_fixed_bytes;
            }
            if (context_fixed_bytes > SIZE_MAX || context_host_bytes > SIZE_MAX) {
                throw std::runtime_error("MoE cache context allocation exceeds address space");
            }
            add(pimpl->moe_cache_memory[entry.first].fixed_device_bytes,
                static_cast<size_t>(context_fixed_bytes));
            add(pimpl->moe_cache_context_memory[context_index][entry.first].fixed_device_bytes,
                static_cast<size_t>(context_fixed_bytes));
            add(pimpl->moe_cache_context_host_memory[context_index], static_cast<size_t>(context_host_bytes));
        }
    }
    if (!unmatched_selected_layers.empty()) {
        const int32_t layer = *std::min_element(unmatched_selected_layers.begin(), unmatched_selected_layers.end());
        throw std::runtime_error(format(
            "MoE cache layer selector includes layer %d, which has no routed expert tensor group", layer));
    }

    if (pimpl->moe_cache_byte_budgets_owned.empty()) {
        for (const auto & entry : pimpl->moe_cache_memory) {
            pimpl->moe_cache_slots[entry.first] = params.moe_expert_cache_slots;
        }
    } else {
        if (pimpl->moe_cache_byte_budgets_owned.size() != 1 &&
                pimpl->moe_cache_byte_budgets_owned.size() != devices.size()) {
            throw std::runtime_error("MoE cache byte budget count must be one or match the selected device count");
        }
        for (size_t i = 0; i < devices.size(); ++i) {
            ggml_backend_dev_t dev = devices[i].dev;
            const auto found = pimpl->moe_cache_memory.find(dev);
            if (found == pimpl->moe_cache_memory.end()) {
                continue;
            }
            const size_t budget = pimpl->moe_cache_byte_budgets_owned.size() == 1 ?
                pimpl->moe_cache_byte_budgets_owned[0] : pimpl->moe_cache_byte_budgets_owned[i];
            const auto & memory = found->second;
            const size_t available = budget > memory.fixed_device_bytes ? budget - memory.fixed_device_bytes : 0;
            const uint64_t slots = memory.per_slot_device_bytes != 0 ?
                std::min<uint64_t>(available / memory.per_slot_device_bytes, memory.max_slots) : 0;
            if (slots == 0) {
                throw std::runtime_error("MoE cache byte budget cannot hold one slot on an active device");
            }
            pimpl->moe_cache_slots[dev] = static_cast<int32_t>(slots);
            LLAMA_LOG_INFO("moe-cache-budget: device=%s budget_bytes=%zu slots=%llu fixed_device_bytes=%zu "
                           "per_slot_device_bytes=%zu\n",
                           ggml_backend_dev_name(dev), budget, (unsigned long long) slots,
                           memory.fixed_device_bytes, memory.per_slot_device_bytes);
        }
    }

    if (params.moe_expert_cache_host_pinned_size > 0) {
        size_t mandatory = 0;
        for (const auto ctx_type : {LLAMA_CONTEXT_TYPE_DEFAULT, LLAMA_CONTEXT_TYPE_MTP}) {
            for (const auto & entry : moe_expert_cache_host_staging(ctx_type)) {
                add(mandatory, entry.second);
            }
        }
        if (mandatory > params.moe_expert_cache_host_pinned_size) {
            throw std::runtime_error(format(
                "MoE cache host-pinned budget cannot hold mandatory staging: need %zu bytes, have %zu",
                mandatory, params.moe_expert_cache_host_pinned_size));
        }
    }
}

void llama_model::build_moe_sources() {
    std::vector<llama_moe_source_group> result;
    result.reserve(layers.size() * 2);
    auto append = [&](uint32_t domain, bool route_present, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * gate_up, ggml_tensor * down) -> llama_moe_source_group * {
        if (!gate && !up && !gate_up && !down) {
            return nullptr;
        }
        uint32_t layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_INVALID;
        if (gate && up && !gate_up && down) {
            layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_SEPARATE;
        } else if (!gate && !up && gate_up && down) {
            layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_FUSED_GATE_UP;
        } else if (!gate && up && !gate_up && down) {
            layout = GGML_BACKEND_MOE_CANDIDATE_LAYOUT_UNGATED;
        }
        result.push_back({layout, domain, route_present, {}});
        auto & banks = result.back().banks;
        ggml_tensor * weights[] = {gate, up, gate_up, down};
        const uint32_t roles[] = {
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_WEIGHT,
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_WEIGHT,
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_WEIGHT,
            GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_WEIGHT,
        };
        for (uint32_t i = 0; i < 4; ++i) {
            if (weights[i]) {
                banks.push_back({weights[i], roles[i], GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE});
            }
        }
        return &result.back();
    };
    for (size_t il = 0; il < layers.size(); ++il) {
        const auto & layer = layers[il];
        auto * group = append(GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_ORDINARY, layer.ffn_gate_inp != nullptr,
            layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_gate_up_exps, layer.ffn_down_exps);
        if (group) {
            group->layer = static_cast<int32_t>(il);
            auto add = [&](ggml_tensor * tensor, uint32_t role, uint32_t status) {
                if (tensor) {
                    group->banks.push_back({tensor, role, status});
                }
            };
            add(layer.ffn_gate_exps_s, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_SCALE, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_SCALE);
            add(layer.ffn_up_exps_s, layer.ffn_gate_up_exps ? GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_SCALE : GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_SCALE, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_SCALE);
            add(layer.ffn_down_exps_s, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_SCALE, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_SCALE);
            add(layer.ffn_gate_exps_b, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_BIAS, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_BIAS);
            add(layer.ffn_up_exps_b, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_BIAS, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_BIAS);
            add(layer.ffn_gate_up_exps_b, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_UP_BIAS, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_BIAS);
            add(layer.ffn_down_exps_b, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_BIAS, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_OUTPUT_BIAS);
            add(layer.ffn_gate_exps_in_s, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_GATE_INPUT_SCALE, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_INPUT_SCALE);
            add(layer.ffn_up_exps_in_s, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_UP_INPUT_SCALE, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_INPUT_SCALE);
            add(layer.ffn_down_exps_in_s, GGML_BACKEND_MOE_CANDIDATE_BANK_ROLE_DOWN_INPUT_SCALE, GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_INPUT_SCALE);
        }
        auto * chunk = append(GGML_BACKEND_MOE_CANDIDATE_DOMAIN_V2_CHUNK, true, layer.ffn_gate_chexps,
                              layer.ffn_up_chexps, nullptr, layer.ffn_down_chexps);
        if (chunk) {
            chunk->layer = static_cast<int32_t>(il);
        }
    }
    pimpl->moe_sources = std::move(result);
}

const std::vector<llama_moe_source_group> & llama_model::moe_sources() const {
    return pimpl->moe_sources;
}

static std::string json_quote(const std::string & value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (ch < 0x20 || ch >= 0x7f) {
                    static const char hex[] = "0123456789abcdef";
                    out << "\\u00" << hex[ch >> 4] << hex[ch & 0x0f];
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    out << '"';
    return out.str();
}

struct moe_device_identity {
    std::string backend;
    std::string canonical_id;
    std::string identity_kind;
    std::string physical_id;
    std::string name;
    std::string description;
};

static moe_device_identity describe_moe_device(ggml_backend_dev_t dev) {
    moe_device_identity result;
    if (dev == nullptr) {
        result.identity_kind = "unavailable";
        return result;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    result.backend = reg != nullptr ? ggml_backend_reg_name(reg) : "";
    result.name = ggml_backend_dev_name(dev);
    result.description = ggml_backend_dev_description(dev);
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(dev, &props);
    result.physical_id = props.device_id != nullptr ? props.device_id : "";
    size_t registry_index = 0;
    const size_t n_devices = ggml_backend_dev_count();
    for (; registry_index < n_devices && ggml_backend_dev_get(registry_index) != dev; ++registry_index) {
    }
    const std::string fallback = result.name + "\n" + result.description + "\nregistry:" +
        (registry_index < n_devices ? std::to_string(registry_index) : "unknown");
    const std::string value = !result.physical_id.empty() ? result.physical_id : fallback;
    result.identity_kind = !result.physical_id.empty() ? "physical" : "runtime_name_unverified";
    result.canonical_id = std::to_string(result.backend.size()) + ":" + result.backend + ":" +
        std::to_string(value.size()) + ":" + value;
    return result;
}

static bool is_moe_cache_source_buft(ggml_backend_buffer_type_t buft) {
    if (buft == nullptr) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto is_cache_buft = reg != nullptr ? reinterpret_cast<ggml_backend_moe_cache_is_buffer_type_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_MOE_CACHE_IS_BUFFER_TYPE_PROC_NAME)) : nullptr;
    return is_cache_buft != nullptr && is_cache_buft(buft);
}

llama_moe_placement_report llama_model::moe_placement() const {
    llama_moe_placement_report report;
    report.uses_mmap = pimpl->resolved_uses_mmap;
    report.direct_io_state = pimpl->resolved_direct_io_state;
    report.uses_mlock = pimpl->resolved_uses_mlock;
    report.has_lazy_tensors = pimpl->resolved_has_lazy_tensors;
    report.model_identity_kind = pimpl->artifact_sources.empty() ? "layout_unverified" : "local_unverified";

    std::ostringstream model_record;
    model_record.imbue(std::locale::classic());
    model_record << "{\"arch\":" << json_quote(arch_name()) << ",\"artifact_sources\":[";
    for (size_t i = 0; i < pimpl->artifact_sources.size(); ++i) {
        if (i != 0) {
            model_record << ',';
        }
        const auto & source = pimpl->artifact_sources[i];
        model_record << "{\"file_size\":" << source.file_size << ",\"index\":" << i
                     << ",\"modification_time\":";
        if (source.modification_time_available) {
            model_record << source.modification_time;
        } else {
            model_record << "null";
        }
        model_record << '}';
    }
    model_record << "],\"ftype\":" << static_cast<int>(ftype()) << ",\"identity_kind\":"
                 << json_quote(report.model_identity_kind) << ",\"moe_groups\":[";
    for (size_t group_index = 0; group_index < pimpl->moe_sources.size(); ++group_index) {
        if (group_index != 0) {
            model_record << ',';
        }
        const auto & group = pimpl->moe_sources[group_index];
        uint32_t n_experts = 0;
        for (const auto & bank : group.banks) {
            if (bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE && bank.tensor != nullptr &&
                bank.tensor->ne[2] > 0 && bank.tensor->ne[2] <= UINT32_MAX) {
                n_experts = static_cast<uint32_t>(bank.tensor->ne[2]);
                break;
            }
        }
        model_record << "{\"banks\":[";
        for (size_t bank_index = 0; bank_index < group.banks.size(); ++bank_index) {
            if (bank_index != 0) {
                model_record << ',';
            }
            const auto & bank   = group.banks[bank_index];
            const auto * tensor = bank.tensor;
            model_record << "{\"name\":" << json_quote(tensor != nullptr ? tensor->name : "");
            if (tensor != nullptr) {
                model_record << ",\"nb\":[" << tensor->nb[0] << ',' << tensor->nb[1] << ',' << tensor->nb[2] << ','
                             << tensor->nb[3] << "],\"ne\":[" << tensor->ne[0] << ',' << tensor->ne[1] << ','
                             << tensor->ne[2] << ',' << tensor->ne[3] << ']';
            }
            model_record << ",\"role\":" << bank.role << ",\"status\":" << bank.status;
            if (tensor != nullptr) {
                model_record << ",\"type\":" << static_cast<int>(tensor->type);
            }
            model_record << '}';
        }
        model_record << "],\"domain\":" << group.domain << ",\"index\":" << group_index << ",\"layer\":" << group.layer
                     << ",\"layout\":" << group.layout << ",\"n_experts\":" << n_experts
                     << ",\"route_present\":" << (group.route_present ? "true" : "false")
                     << ",\"top_k\":"
                     << (group.layer >= 0 ? std::max<uint32_t>(1, hparams.n_expert_used(group.layer)) : 0) << '}';
    }
    struct identity_tensor {
        std::string name;
        int32_t type = 0;
        std::array<int64_t, GGML_MAX_DIMS> ne = {};
        std::array<size_t, GGML_MAX_DIMS> nb = {};
    };
    std::vector<identity_tensor> ordered_tensors;
    ordered_tensors.reserve(tensors_by_name.size() + pimpl->borrowed_tensors.size());
    std::unordered_set<std::string> identity_tensor_names;
    for (const auto & tensor : tensors_by_name) {
        identity_tensor identity;
        identity.name = tensor.first;
        identity.type = static_cast<int32_t>(tensor.second->type);
        std::copy(std::begin(tensor.second->ne), std::end(tensor.second->ne), identity.ne.begin());
        std::copy(std::begin(tensor.second->nb), std::end(tensor.second->nb), identity.nb.begin());
        ordered_tensors.push_back(std::move(identity));
        identity_tensor_names.insert(tensor.first);
    }
    for (const auto & [name, tensor] : pimpl->borrowed_tensors) {
        if (identity_tensor_names.count(name) != 0) {
            continue;
        }
        identity_tensor identity;
        identity.name = name;
        identity.type = tensor.type;
        identity.ne = tensor.ne;
        identity.nb = tensor.nb;
        ordered_tensors.push_back(std::move(identity));
    }
    std::sort(ordered_tensors.begin(), ordered_tensors.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.name < rhs.name;
    });
    model_record << "],\"tensor_count\":" << ordered_tensors.size() << ",\"tensors\":[";
    for (size_t i = 0; i < ordered_tensors.size(); ++i) {
        if (i != 0) {
            model_record << ',';
        }
        const auto & tensor = ordered_tensors[i];
        model_record << "{\"name\":" << json_quote(tensor.name) << ",\"nb\":[" << tensor.nb[0] << ','
                     << tensor.nb[1] << ',' << tensor.nb[2] << ',' << tensor.nb[3] << "],\"ne\":["
                     << tensor.ne[0] << ',' << tensor.ne[1] << ',' << tensor.ne[2] << ',' << tensor.ne[3]
                     << "],\"type\":" << tensor.type << '}';
    }
    model_record << "]}";
    const std::string model_canonical = model_record.str();
    report.model_identity_record      = model_canonical;
    report.model_identity             = hash_sha256_hex(model_canonical.data(), model_canonical.size());

    std::vector<std::string> shared_names;
    shared_names.reserve(pimpl->borrowed_tensors.size());
    for (const auto & entry : pimpl->borrowed_tensors) {
        shared_names.push_back(entry.first);
    }
    std::sort(shared_names.begin(), shared_names.end());
    for (const auto & shared_name : shared_names) {
        llama_moe_shared_tensor shared;
        shared.name = shared_name;
        shared = pimpl->borrowed_tensors.at(shared_name);
        report.shared_tensors.push_back(std::move(shared));
    }

    const auto owner_index = [&](ggml_backend_dev_t dev) -> int32_t {
        for (size_t i = 0; i < devices.size(); ++i) {
            if (devices[i].dev == dev) {
                return static_cast<int32_t>(i);
            }
        }
        return -1;
    };
    const auto owner_index_by_canonical_id = [&](const std::string & canonical_id) -> int32_t {
        for (size_t i = 0; i < report.owners.size(); ++i) {
            if (report.owners[i].canonical_id == canonical_id) {
                return static_cast<int32_t>(i);
            }
        }
        return -1;
    };
    const auto staging_default = moe_expert_cache_host_staging(LLAMA_CONTEXT_TYPE_DEFAULT);
    const auto staging_mtp = moe_expert_cache_host_staging(LLAMA_CONTEXT_TYPE_MTP);
    for (size_t i = 0; i < devices.size(); ++i) {
        ggml_backend_dev_t     dev   = devices[i].dev;
        const auto identity = describe_moe_device(dev);
        llama_moe_placement_owner owner;
        owner.selected_index = static_cast<uint32_t>(i);
        owner.id             = identity.physical_id;
        owner.name           = identity.name;
        owner.description    = identity.description;
        owner.backend        = identity.backend;
        owner.canonical_id   = identity.canonical_id;
        owner.identity_kind  = identity.identity_kind;
        owner.slots          = moe_expert_cache_slots(dev);
        if (!pimpl->moe_cache_byte_budgets_owned.empty()) {
            owner.cache_byte_cap = pimpl->moe_cache_byte_budgets_owned.size() == 1 ?
                                       pimpl->moe_cache_byte_budgets_owned[0] :
                                       pimpl->moe_cache_byte_budgets_owned.at(i);
        }
        const auto found = pimpl->moe_cache_memory.find(dev);
        if (found != pimpl->moe_cache_memory.end()) {
            owner.cache_fixed_bytes    = found->second.fixed_device_bytes;
            owner.cache_per_slot_bytes = found->second.per_slot_device_bytes;
        }
        const auto found_staging_default = staging_default.find(dev);
        const auto found_staging_mtp = staging_mtp.find(dev);
        owner.mandatory_host_staging_default_bytes = found_staging_default != staging_default.end() ?
            found_staging_default->second : 0;
        owner.mandatory_host_staging_mtp_bytes = found_staging_mtp != staging_mtp.end() ?
            found_staging_mtp->second : 0;
        owner.mandatory_host_staging_available = params.moe_expert_cache_host_pinned_size > 0;
        if (owner.mandatory_host_staging_available) {
            owner.mandatory_host_staging_provenance = "backend_staging_size_v1_exact";
        }
        if (owner.mandatory_host_staging_default_bytes >
                SIZE_MAX - owner.mandatory_host_staging_mtp_bytes ||
            owner.mandatory_host_staging_default_bytes + owner.mandatory_host_staging_mtp_bytes >
                SIZE_MAX - report.mandatory_host_staging_bytes) {
            throw std::overflow_error("MoE placement host staging accounting overflow");
        }
        report.mandatory_host_staging_bytes +=
            owner.mandatory_host_staging_default_bytes + owner.mandatory_host_staging_mtp_bytes;
        report.owners.push_back(std::move(owner));
    }

    // This is the authoritative whole-buffer model-weight ledger. Per-group
    // attribution below is intentionally diagnostic because packed buffers,
    // views, and aliases are not generally additive.
    for (const auto & [ctx, bufs] : pimpl->ctxs_bufs) {
        for (const auto & buffer : bufs) {
            if (buffer == nullptr) {
                continue;
            }
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buffer.get());
            const bool cache_source = is_moe_cache_source_buft(buft);
            ggml_backend_dev_t dev = cache_source ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) :
                                                    ggml_backend_buft_get_device(buft);
            const auto identity = describe_moe_device(dev);
            const bool current = !params.no_alloc;
            const size_t bytes = current ? ggml_backend_buffer_get_size(buffer.get()) :
                ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
            const std::string resolved_class = cache_source ? "moe_cache_source_host" :
                ggml_backend_buft_is_host(buft) ? "ordinary_host" :
                "ordinary_" + identity.backend + "_device";
            auto found = std::find_if(
                report.model_allocations.begin(), report.model_allocations.end(), [&](const auto & allocation) {
                    return allocation.resolved_class == resolved_class &&
                           allocation.owner_canonical_id == identity.canonical_id &&
                           allocation.current_allocation == current;
                });
            if (found == report.model_allocations.end()) {
                llama_moe_model_allocation allocation;
                allocation.resolved_class = resolved_class;
                allocation.owner_canonical_id = identity.canonical_id;
                allocation.owner_identity_kind = identity.identity_kind;
                allocation.owner_backend = identity.backend;
                allocation.bytes_available = true;
                allocation.current_allocation = current;
                allocation.provenance = current ? "backend_packed_buffer_current" :
                                                  "backend_packed_context_size_estimate";
                report.model_allocations.push_back(std::move(allocation));
                found = std::prev(report.model_allocations.end());
            }
            if (bytes > SIZE_MAX - found->bytes) {
                throw std::overflow_error("MoE placement model allocation accounting overflow");
            }
            found->bytes += bytes;
        }
    }

    std::unordered_set<const ggml_tensor *> counted_ordinary_roots;
    for (size_t group_index = 0; group_index < pimpl->moe_sources.size(); ++group_index) {
        const auto &              source = pimpl->moe_sources[group_index];
        llama_moe_placement_group group;
        group.semantic_index = static_cast<uint32_t>(group_index);
        group.layer          = source.layer;
        group.layout         = source.layout;
        group.domain         = source.domain;
        group.route_present  = source.route_present;
        group.top_k          = source.layer >= 0 ? std::max<uint32_t>(1, hparams.n_expert_used(source.layer)) : 0;
        group.context_use_mask = llama_moe_placement_context_mask(
            params.load_mtp, hparams.n_layer_nextn, hparams.router_layer, hparams.n_layer(), source.layer);

        bool base_placement_set = false;
        llama_moe_placement_mode base_mode = LLAMA_MOE_PLACEMENT_ORDINARY_CPU;
        llama_moe_placement_reason base_reason = LLAMA_MOE_PLACEMENT_DEFAULT;
        std::string base_owner_canonical_id;
        std::string base_resolved_class;
        bool mixed_reason = false;
        for (const auto & source_bank : source.banks) {
            const auto *             tensor = source_bank.tensor;
            llama_moe_placement_bank bank;
            bank.role   = source_bank.role;
            bank.status = source_bank.status;
            if (tensor != nullptr) {
                bank.name          = tensor->name;
                bank.type          = static_cast<int32_t>(tensor->type);
                bank.expert_stride = tensor->nb[2];
                bank.tensor_bytes  = ggml_nbytes(tensor);
                ggml_backend_buffer_type_t buft = tensor->buffer != nullptr ?
                    ggml_backend_buffer_get_type(tensor->buffer) : nullptr;
                const auto cached_tensor = pimpl->moe_cache_tensor_owners.find(tensor);
                const bool cached = cached_tensor != pimpl->moe_cache_tensor_owners.end();
                ggml_backend_dev_t owner = cached ? cached_tensor->second :
                    (buft != nullptr ? ggml_backend_buft_get_device(buft) : nullptr);
                const auto resolution = pimpl->tensor_override_resolutions.find(tensor->name);
                bank.actual_buft_available = !params.no_alloc && buft != nullptr;
                bank.actual_buft = bank.actual_buft_available ? ggml_backend_buft_name(buft) : "";
                bank.selected_buft = bank.actual_buft;
                bank.resolved_buft = bank.actual_buft;
                if (cached) {
                    bank.mode = LLAMA_MOE_PLACEMENT_RESIDUAL_CACHE;
                    bank.resolved_class = "residual_cache";
                } else if (resolution != pimpl->tensor_override_resolutions.end()) {
                    bank.mode = resolution->second.resolved_is_host ? LLAMA_MOE_PLACEMENT_ORDINARY_CPU :
                                                                     LLAMA_MOE_PLACEMENT_ORDINARY_DEVICE;
                    bank.resolved_class = resolution->second.resolved_is_host ? "ordinary_host" :
                        "ordinary_" + resolution->second.resolved_owner_backend + "_device";
                } else {
                    bank.mode = buft != nullptr && ggml_backend_buft_is_host(buft) ?
                        LLAMA_MOE_PLACEMENT_ORDINARY_CPU : LLAMA_MOE_PLACEMENT_ORDINARY_DEVICE;
                    bank.resolved_class = bank.mode == LLAMA_MOE_PLACEMENT_ORDINARY_CPU ?
                        "ordinary_host" : "ordinary_device_unverified";
                }
                if (owner != nullptr) {
                    const auto identity = describe_moe_device(owner);
                    bank.owner_id = identity.physical_id;
                    bank.owner_name = identity.name;
                    bank.owner_canonical_id = identity.canonical_id;
                    bank.owner_identity_kind = identity.identity_kind;
                    bank.owner_backend = identity.backend;
                }
                if (resolution != pimpl->tensor_override_resolutions.end()) {
                    switch (resolution->second.origin) {
                        case llama_model_loader::TENSOR_OVERRIDE_DEFAULT:
                            bank.reason = LLAMA_MOE_PLACEMENT_DEFAULT;
                            break;
                        case llama_model_loader::TENSOR_OVERRIDE_USER:
                            bank.reason = LLAMA_MOE_PLACEMENT_USER_OVERRIDE;
                            break;
                        case llama_model_loader::TENSOR_OVERRIDE_CACHE_LEGACY:
                            bank.reason = LLAMA_MOE_PLACEMENT_CACHE_LEGACY;
                            break;
                        case llama_model_loader::TENSOR_OVERRIDE_CACHE_SELECTOR:
                            bank.reason = LLAMA_MOE_PLACEMENT_CACHE_SELECTOR;
                            break;
                        default:
                            throw std::runtime_error("invalid MoE placement override origin");
                    }
                    bank.winning_override_index = static_cast<int32_t>(resolution->second.index);
                    bank.winning_override_pattern = resolution->second.pattern;
                    bank.requested_buft = resolution->second.requested_buft;
                    bank.selected_buft = resolution->second.selected_buft;
                    bank.resolved_buft = resolution->second.resolved_buft;
                    if (bank.owner_canonical_id.empty()) {
                        bank.owner_canonical_id = resolution->second.resolved_owner_canonical_id;
                        bank.owner_identity_kind = resolution->second.resolved_owner_identity_kind;
                        bank.owner_backend = resolution->second.resolved_owner_backend;
                    }
                }
                bank.owner_index = owner != nullptr ? owner_index(owner) :
                    owner_index_by_canonical_id(bank.owner_canonical_id);
                if (cached) {
                    if (group.cache_owner_canonical_id.empty()) {
                        group.cache_owner_canonical_id = bank.owner_canonical_id;
                        group.cache_owner_index = bank.owner_index;
                    } else if (group.cache_owner_canonical_id != bank.owner_canonical_id) {
                        throw std::runtime_error("MoE cache group spans multiple physical owners");
                    }
                }
                if (!cached && tensor->buffer != nullptr) {
                    bank.allocation_estimate =
                        ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(tensor->buffer), tensor);
                    bank.allocation_estimate_available = true;
                    bank.allocation_provenance = "buffer_type_alloc_size_estimate";
                }
                if (source_bank.status == GGML_BACKEND_MOE_CANDIDATE_STATUS_V2_ROUTED_BASE) {
                    if (tensor->ne[2] > 0 && tensor->ne[2] <= UINT32_MAX) {
                        group.n_experts = static_cast<uint32_t>(tensor->ne[2]);
                    }
                }
                // A semantic group includes all of its typed banks, not only
                // the routed bases. Auxiliary scales/biases may resolve to a
                // different class or owner and must make the group mixed.
                if (!base_placement_set) {
                    base_placement_set = true;
                    base_mode = bank.mode;
                    base_reason = bank.reason;
                    base_owner_canonical_id = bank.owner_canonical_id;
                    base_resolved_class = bank.resolved_class;
                } else {
                    if (base_mode != bank.mode || base_owner_canonical_id != bank.owner_canonical_id ||
                        base_resolved_class != bank.resolved_class) {
                        group.mode = LLAMA_MOE_PLACEMENT_MIXED;
                    }
                    mixed_reason = mixed_reason || base_reason != bank.reason;
                }
            }
            group.banks.push_back(std::move(bank));
        }
        if (group.mode != LLAMA_MOE_PLACEMENT_MIXED) {
            group.mode = base_mode;
        }
        group.placement_reason = mixed_reason ? "mixed" :
            (base_reason == LLAMA_MOE_PLACEMENT_USER_OVERRIDE ? "user_override" :
             base_reason == LLAMA_MOE_PLACEMENT_CACHE_LEGACY ? "legacy_cache" :
             base_reason == LLAMA_MOE_PLACEMENT_CACHE_SELECTOR ? "cache_selector" : "default");
        group.owner_index = group.mode == LLAMA_MOE_PLACEMENT_MIXED ? -1 :
            owner_index_by_canonical_id(base_owner_canonical_id);
        if (group.mode != LLAMA_MOE_PLACEMENT_MIXED) {
            group.owner_canonical_id = base_owner_canonical_id;
            if (group.owner_index >= 0) {
                const auto & owner = report.owners[group.owner_index];
                group.owner_id = owner.id;
                group.owner_name = owner.name;
            }
        }
        {
            for (size_t bank_index = 0; bank_index < source.banks.size(); ++bank_index) {
                const auto & source_bank = source.banks[bank_index];
                const auto * tensor = source_bank.tensor;
                if (tensor == nullptr || tensor->buffer == nullptr ||
                    group.banks[bank_index].mode == LLAMA_MOE_PLACEMENT_RESIDUAL_CACHE) {
                    continue;
                }
                const ggml_tensor * root = tensor;
                while (root->view_src != nullptr) {
                    root = root->view_src;
                }
                if (!counted_ordinary_roots.insert(root).second) {
                    continue;
                }
                const size_t bytes =
                    ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(root->buffer), root);
                if (bytes > SIZE_MAX - group.ordinary_allocation_estimate) {
                    throw std::overflow_error("MoE placement allocation estimate overflow");
                }
                group.ordinary_allocation_estimate += bytes;
                const int32_t bank_owner = group.banks[bank_index].owner_index;
                if (bank_owner >= 0 && static_cast<size_t>(bank_owner) < report.owners.size()) {
                    auto & ordinary = report.owners[bank_owner].ordinary_allocation_estimate;
                    if (bytes > SIZE_MAX - ordinary) {
                        throw std::overflow_error("MoE placement owner allocation estimate overflow");
                    }
                    ordinary += bytes;
                }
            }
            group.ordinary_allocation_provenance = "buffer_type_alloc_size_unique_tensor_roots_estimate";
        }
        if (group_index < pimpl->moe_cache_group_context_memory[0].size()) {
            const auto & default_memory = pimpl->moe_cache_group_context_memory[0][group_index];
            const auto & mtp_memory = pimpl->moe_cache_group_context_memory[1][group_index];
            group.cache_fixed_default_bytes = default_memory.fixed_device_bytes;
            group.cache_per_slot_default_bytes = default_memory.per_slot_device_bytes;
            group.cache_fixed_mtp_bytes = mtp_memory.fixed_device_bytes;
            group.cache_per_slot_mtp_bytes = mtp_memory.per_slot_device_bytes;
            if (group.cache_fixed_mtp_bytes > SIZE_MAX - group.cache_fixed_default_bytes ||
                group.cache_per_slot_mtp_bytes > SIZE_MAX - group.cache_per_slot_default_bytes) {
                throw std::overflow_error("MoE placement group accounting overflow");
            }
            group.cache_fixed_bytes = group.cache_fixed_default_bytes + group.cache_fixed_mtp_bytes;
            group.cache_per_slot_bytes = group.cache_per_slot_default_bytes + group.cache_per_slot_mtp_bytes;
            if (group.cache_fixed_bytes != 0 || group.cache_per_slot_bytes != 0) {
                group.cache_sizing_provenance = "backend_size_query_sum_by_context";
            }
        }
        report.groups.push_back(std::move(group));
    }
    for (const auto & group : report.groups) {
        std::unordered_set<int32_t> active_owners;
        for (const auto & bank : group.banks) {
            if (bank.owner_index >= 0 && static_cast<size_t>(bank.owner_index) < report.owners.size()) {
                active_owners.insert(bank.owner_index);
            }
        }
        for (const int32_t index : active_owners) {
            auto & owner = report.owners[index];
            ++owner.active_groups;
            owner.active_context_mask |= group.context_use_mask;
        }
        if ((group.cache_fixed_bytes == 0 && group.cache_per_slot_bytes == 0) || group.cache_owner_index < 0 ||
            static_cast<size_t>(group.cache_owner_index) >= report.owners.size()) {
            continue;
        }
        auto & owner = report.owners[group.cache_owner_index];
        ++owner.active_cache_groups;
        owner.active_cache_context_mask |= group.context_use_mask;
        if (group.cache_fixed_bytes > SIZE_MAX - owner.cache_group_fixed_bytes) {
            throw std::overflow_error("MoE placement owner accounting overflow");
        }
        owner.cache_group_fixed_bytes += group.cache_fixed_bytes;
        if (group.cache_per_slot_bytes > SIZE_MAX - owner.cache_group_per_slot_bytes) {
            throw std::overflow_error("MoE placement owner accounting overflow");
        }
        owner.cache_group_per_slot_bytes += group.cache_per_slot_bytes;
    }
    for (auto & owner : report.owners) {
        if (owner.cache_group_fixed_bytes > owner.cache_fixed_bytes) {
            throw std::runtime_error("MoE placement group accounting exceeds owner ledger");
        }
        if (owner.cache_group_per_slot_bytes != owner.cache_per_slot_bytes) {
            throw std::runtime_error("MoE placement per-slot accounting disagrees with owner ledger");
        }
        owner.cache_context_fixed_bytes = owner.cache_fixed_bytes - owner.cache_group_fixed_bytes;
    }

    bool placement_identity_unverified = std::any_of(report.owners.begin(), report.owners.end(), [](const auto & owner) {
        return owner.identity_kind != "physical";
    });
    placement_identity_unverified = placement_identity_unverified ||
        std::any_of(report.shared_tensors.begin(), report.shared_tensors.end(), [](const auto & shared) {
            return shared.resolved_storage_available && shared.owner_identity_kind != "physical";
        });
    for (const auto & group : report.groups) {
        placement_identity_unverified = placement_identity_unverified ||
            std::any_of(group.banks.begin(), group.banks.end(), [](const auto & bank) {
                return !bank.owner_canonical_id.empty() && bank.owner_identity_kind != "physical";
            });
    }
    report.placement_identity_kind = placement_identity_unverified ? "runtime_unverified" : "physical";

    std::ostringstream groups_record;
    groups_record.imbue(std::locale::classic());
    groups_record << '[';
    for (size_t i = 0; i < report.groups.size(); ++i) {
        if (i != 0) {
            groups_record << ',';
        }
        const auto & group = report.groups[i];
        groups_record << "{\"banks\":[";
        for (size_t bank_index = 0; bank_index < group.banks.size(); ++bank_index) {
            if (bank_index != 0) {
                groups_record << ',';
            }
            const auto & bank = group.banks[bank_index];
            groups_record << "{\"expert_stride\":" << bank.expert_stride
                          << ",\"mode\":" << static_cast<int>(bank.mode)
                          << ",\"name\":" << json_quote(bank.name)
                          << ",\"owner_canonical_id\":" << json_quote(bank.owner_canonical_id)
                          << ",\"owner_identity_kind\":" << json_quote(bank.owner_identity_kind)
                          << ",\"resolved_class\":" << json_quote(bank.resolved_class)
                          << ",\"role\":" << bank.role
                          << ",\"status\":" << bank.status
                          << ",\"tensor_bytes\":" << bank.tensor_bytes << ",\"type\":" << bank.type << '}';
        }
        groups_record << "],\"cache_fixed_bytes\":" << group.cache_fixed_bytes
                      << ",\"cache_fixed_default_bytes\":" << group.cache_fixed_default_bytes
                      << ",\"cache_fixed_mtp_bytes\":" << group.cache_fixed_mtp_bytes
                      << ",\"cache_owner_canonical_id\":" << json_quote(group.cache_owner_canonical_id)
                      << ",\"cache_per_slot_bytes\":" << group.cache_per_slot_bytes
                      << ",\"cache_per_slot_default_bytes\":" << group.cache_per_slot_default_bytes
                      << ",\"cache_per_slot_mtp_bytes\":" << group.cache_per_slot_mtp_bytes
                      << ",\"cache_sizing_provenance\":" << json_quote(group.cache_sizing_provenance)
                      << ",\"context_use_mask\":" << group.context_use_mask << ",\"domain\":" << group.domain
                      << ",\"index\":" << group.semantic_index << ",\"layer\":" << group.layer
                      << ",\"layout\":" << group.layout << ",\"mode\":" << static_cast<int>(group.mode)
                      << ",\"n_experts\":" << group.n_experts
                      << ",\"owner_canonical_id\":" << json_quote(group.owner_canonical_id)
                      << ",\"route_present\":" << (group.route_present ? "true" : "false")
                      << ",\"top_k\":" << group.top_k << '}';
    }
    groups_record << ']';
    std::ostringstream owners_record;
    owners_record.imbue(std::locale::classic());
    owners_record << '[';
    for (size_t i = 0; i < report.owners.size(); ++i) {
        if (i != 0) {
            owners_record << ',';
        }
        const auto & owner = report.owners[i];
        owners_record << "{\"active_cache_context_mask\":" << owner.active_cache_context_mask
                      << ",\"active_cache_groups\":" << owner.active_cache_groups
                      << ",\"active_context_mask\":" << owner.active_context_mask
                      << ",\"active_groups\":" << owner.active_groups
                      << ",\"backend\":" << json_quote(owner.backend)
                      << ",\"cache_byte_cap\":" << owner.cache_byte_cap
                      << ",\"cache_context_fixed_bytes\":" << owner.cache_context_fixed_bytes
                      << ",\"cache_fixed_bytes\":" << owner.cache_fixed_bytes
                      << ",\"cache_group_fixed_bytes\":" << owner.cache_group_fixed_bytes
                      << ",\"cache_group_per_slot_bytes\":" << owner.cache_group_per_slot_bytes
                      << ",\"cache_per_slot_bytes\":" << owner.cache_per_slot_bytes
                      << ",\"canonical_id\":" << json_quote(owner.canonical_id)
                      << ",\"identity_kind\":" << json_quote(owner.identity_kind)
                      << ",\"selected_index\":" << owner.selected_index
                      << ",\"slots\":" << owner.slots << '}';
    }
    owners_record << ']';
    std::ostringstream shared_record;
    shared_record.imbue(std::locale::classic());
    shared_record << '[';
    for (size_t i = 0; i < report.shared_tensors.size(); ++i) {
        if (i != 0) {
            shared_record << ',';
        }
        const auto & shared = report.shared_tensors[i];
        shared_record << "{\"name\":" << json_quote(shared.name)
                      << ",\"nb\":[" << shared.nb[0] << ',' << shared.nb[1] << ',' << shared.nb[2] << ','
                      << shared.nb[3] << "],\"ne\":[" << shared.ne[0] << ',' << shared.ne[1] << ','
                      << shared.ne[2] << ',' << shared.ne[3] << ']'
                      << ",\"owner_canonical_id\":" << json_quote(shared.owner_canonical_id)
                      << ",\"owner_identity_kind\":" << json_quote(shared.owner_identity_kind)
                      << ",\"resolved_class\":" << json_quote(shared.resolved_class)
                      << ",\"storage_relation\":" << json_quote(shared.storage_relation)
                      << ",\"tensor_bytes\":" << shared.tensor_bytes
                      << ",\"type\":" << shared.type << '}';
    }
    shared_record << ']';
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical << "{\"build\":" << json_quote(LLAMA_COMMIT)
              << ",\"direct_io_state\":" << json_quote(report.direct_io_state)
              << ",\"groups\":" << groups_record.str()
              << ",\"has_lazy_tensors\":" << (report.has_lazy_tensors ? "true" : "false")
              << ",\"host_pinned_bytes\":" << params.moe_expert_cache_host_pinned_size
              << ",\"load_mtp\":" << (params.load_mtp ? "true" : "false")
              << ",\"model_identity\":" << json_quote(report.model_identity)
              << ",\"model_identity_kind\":" << json_quote(report.model_identity_kind)
              << ",\"no_host\":" << (params.no_host ? "true" : "false")
              << ",\"owners\":" << owners_record.str()
              << ",\"placement_identity_kind\":" << json_quote(report.placement_identity_kind)
              << ",\"schema_version\":" << report.schema_version
              << ",\"shared_tensors\":" << shared_record.str()
              << ",\"split_mode\":" << static_cast<int>(params.split_mode)
              << ",\"use_extra_bufts\":" << (params.use_extra_bufts ? "true" : "false")
              << ",\"uses_mlock\":" << (report.uses_mlock ? "true" : "false")
              << ",\"uses_mmap\":" << (report.uses_mmap ? "true" : "false") << '}';
    report.placement_record = canonical.str();
    report.placement_id = hash_sha256_hex(report.placement_record.data(), report.placement_record.size());
    return report;
}

llama_moe_placement_report llama_model_moe_placement(const llama_model * model) {
    return model != nullptr ? model->moe_placement() : llama_moe_placement_report{};
}

const ggml_tensor * llama_model::get_tensor(const char * name) const {
    auto it = std::find_if(tensors_by_name.begin(), tensors_by_name.end(),
            [name](const std::pair<std::string, ggml_tensor *> & it) {
                return it.first == name;
            });
    if (it == tensors_by_name.end()) {
        return nullptr;
    }

    return it->second;
}

float llama_model::get_rope_freq_base (const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_base_train_swa : cparams.rope_freq_base;
}

float llama_model::get_rope_freq_scale(const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
}

ggml_tensor * llama_model::get_rope_factors(const llama_cparams & cparams, int il) const {
    const uint32_t n_ctx_seq = cparams.n_ctx_seq;

    // choose long/short freq factors based on the context size
    if (layers[il].rope_freqs != nullptr) {
        return layers[il].rope_freqs;
    }

    if (n_ctx_seq > hparams.n_ctx_orig_yarn) {
        return layers[il].rope_long;
    }

    return layers[il].rope_short;
}

llama_memory_i * llama_model::create_memory(const llama_memory_params & params, const llama_cparams & cparams) const {
    llama_memory_i * res;
    const llama_memory_placement_options placement = {
        cparams.kv_cpu_pinned,
        cparams.kv_gpu_layers,
        cparams.offload_kqv || cparams.recurrent_state_offload,
    };
    llama_memory_placement_options specialized_placement = placement;
    specialized_placement.gpu_resident_layers = 0;

    switch (arch) {
        // Models that need specific instantiation should be handled in the
        // switch statement
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_JINA_BERT_V3:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_EUROBERT:
        case LLM_ARCH_WAVTOKENIZER_DEC:
        case LLM_ARCH_MODERN_BERT:
        case LLM_ARCH_GEMMA_EMBEDDING:
        case LLM_ARCH_DREAM:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_RND1:
            {
                res = nullptr;
            } break;
        case LLM_ARCH_MINIMAX_M3:
            {
                // sparse (MSA) layers carry an indexer key cache, but leading dense layers do not
                llama_kv_cache::layer_filter_cb filter_idx =
                    [&](int32_t il) { return (uint32_t) il >= hparams.n_layer_dense_lead; };

                res = new llama_kv_cache_msa(
                        *this,
                        params.type_k,
                        params.type_v,
                        !cparams.flash_attn,
                        cparams.offload_kqv,
                        cparams.kv_unified,
                        cparams.n_ctx_seq,
                        cparams.n_seq_max,
                        1,
                        hparams.n_swa,
                        hparams.swa_type,
                        nullptr,
                        filter_idx,
                        nullptr,
                        specialized_placement);
            } break;
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_DEEPSEEK32:
            {
                if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP && hparams.n_layer_nextn > 0) {
                    // The NextN/MTP draft head runs dense MLA (no DSA indexer), so the
                    // MTP context uses a plain attention KV cache holding only the
                    // nextn layer(s) - same pattern as the hybrid Qwen3.5 MTP context.
                    llama_kv_cache::layer_filter_cb filter =
                        [&](uint32_t il) { return il >= hparams.n_layer(); };

                    res = new llama_kv_cache(
                            *this,
                            hparams,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            1,
                            hparams.n_swa,
                            hparams.swa_type,
                            nullptr,
                            filter,
                            nullptr,
                            nullptr,
                            placement);
                } else {
                    // Main context: DSA cache for the trunk layers only - the nextn
                    // layer(s) are never attended by the trunk graph.
                    llama_kv_cache::layer_filter_cb filter_mla = nullptr;
                    if (hparams.n_layer_nextn > 0) {
                        filter_mla = [&](uint32_t il) { return il < hparams.n_layer(); };
                    }
                    llama_kv_cache::layer_filter_cb filter_lid = [&](uint32_t il) { return il < hparams.n_layer() && (arch != LLM_ARCH_GLM_DSA || hparams.is_indexer_full(il)); };

                    res = new llama_kv_cache_dsa(
                            *this,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            1,
                            hparams.n_swa,
                            hparams.swa_type,
                            filter_mla,
                            filter_lid,
                            nullptr,
                            specialized_placement);
                }
            } break;
        case LLM_ARCH_HY_V4:
            {
                if (hparams.indexer_top_k == 0) {
                    // full-attention checkpoint: no indexer, so no indexer key cache
                    res = new llama_kv_cache(
                            *this,
                            hparams,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            1,
                            hparams.n_swa,
                            hparams.swa_type,
                            nullptr,
                            nullptr,
                            nullptr,
                            nullptr,
                            placement);
                } else {
                    // only "full" layers own an indexer, so the shared layers need no indexer cache
                    llama_kv_cache::layer_filter_cb filter_lid = [&](uint32_t il) { return hparams.is_indexer_full(il); };

                    res = new llama_kv_cache_dsa(
                            *this,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            1,
                            hparams.n_swa,
                            hparams.swa_type,
                            nullptr,
                            filter_lid,
                            nullptr,
                            specialized_placement);
                }
            } break;
        case LLM_ARCH_DOTS3NOTE:
            {
                GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE);

                if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP && hparams.n_layer_nextn > 0) {
                    // MTP draft context: plain attention KV cache holding only the nextn layer
                    llama_kv_cache::layer_filter_cb filter =
                        [&](uint32_t il) { return il >= hparams.n_layer(); };

                    res = new llama_kv_cache(
                            *this,
                            hparams,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            1,
                            hparams.n_swa,
                            hparams.swa_type,
                            nullptr,
                            filter,
                            nullptr,
                            nullptr,
                            placement);
                } else {
                    // main context: DSA cache for the trunk full-attention layers plus a window-sized SWA cache
                    llama_kv_cache::layer_filter_cb filter_mla = nullptr;
                    if (hparams.n_layer_nextn > 0) {
                        filter_mla = [&](uint32_t il) { return il < hparams.n_layer(); };
                    }
                    llama_kv_cache::layer_filter_cb filter_lid = [&](uint32_t il) { return il < hparams.n_layer() && hparams.is_indexer_full(il); };

                    res = new llama_kv_cache_dsa_iswa(
                            *this,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            params.swa_full,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            cparams.n_ubatch,
                            1,
                            filter_mla,
                            filter_lid,
                            nullptr,
                            specialized_placement);
                }
            } break;
        case LLM_ARCH_DEEPSEEK4:
            {
                GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE);

                if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
                    const llama_memory_i::layer_filter_cb filter_mtp = [&](int32_t il) {
                        return il >= (int32_t) hparams.n_layer();
                    };

                    res = new llama_kv_cache_iswa(
                            *this,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            params.swa_full,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            cparams.n_ubatch,
                            1,
                            nullptr,
                            filter_mtp,
                            nullptr,
                            nullptr,
                            specialized_placement);
                } else {
                    res = new llama_kv_cache_dsv4(
                            *this,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            params.swa_full,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            cparams.n_ubatch,
                            1,
                            cparams.n_rs_seq,
                            nullptr,
                            nullptr,
                            specialized_placement);
                }
            } break;
        case LLM_ARCH_DFLASH:
            {
                // DSV4 DSpark stages store a single MLA-style K per position (window = the draft ring)
                if (hparams.dsv4_hc_mult > 0) {
                    GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE);

                    res = new llama_kv_cache_iswa(
                            *this,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            params.swa_full,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            cparams.n_ubatch,
                            1,
                            nullptr,
                            nullptr,
                            nullptr,
                            nullptr,
                            specialized_placement);
                    break;
                }
            }
            [[fallthrough]];
        // Models that need standard caching should rely on recurrent/hybrid
        // checks
        default:
            {
                // Dense MTP heads use a plain attention KV cache instead of the hybrid wrapper.
                const bool mtp_on_hybrid_qwen =
                    params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
                    (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE ||
                     arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_QWEN4EXP);

                const bool mtp_on_hybrid_nemotron =
                    params.ctx_type == LLAMA_CONTEXT_TYPE_MTP && arch == LLM_ARCH_NEMOTRON_H_MOE;

                if (llm_arch_is_recurrent(arch)) {
                    res = new llama_memory_recurrent(
                            *this,
                            GGML_TYPE_F32,
                            GGML_TYPE_F32,
                            placement.recurrent_offload,
                            std::max((uint32_t) 1, cparams.n_seq_max),
                            cparams.n_seq_max,
                            cparams.n_rs_seq,
                            nullptr);
                } else if (llm_arch_is_hybrid(arch) && !mtp_on_hybrid_qwen && !mtp_on_hybrid_nemotron) {
                    // The main difference between hybrid architectures is the
                    // layer filters, so pick the right one here
                    llama_memory_hybrid::layer_filter_cb filter_attn = nullptr;
                    llama_memory_hybrid::layer_filter_cb filter_recr = nullptr;
                    // only the sparse-attention architectures use llama_memory_hybrid_idx
                    // a null filter_idx means the GGUF has no indexer tensors
                    llama_memory_hybrid::layer_filter_cb filter_idx  = nullptr;
                    const bool needs_mem_idx = (arch == LLM_ARCH_QWEN4EXP);
                    if (arch == LLM_ARCH_FALCON_H1) {
                        filter_attn = [&](uint32_t) { return true; };
                        filter_recr = [&](uint32_t) { return true; };
                    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
                        filter_attn = [&](uint32_t il) {
                            return !hparams.is_recr(il) && hparams.n_ff(il) == 0;
                        };
                        filter_recr = [&](uint32_t il) {
                            return hparams.is_recr(il) && hparams.n_ff(il) == 0;
                        };
                    } else if (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_MINIMAX_01) {
                        filter_attn = [&](uint32_t il) {
                            return il < hparams.n_layer() && !hparams.is_recr(il);
                        };
                        filter_recr = [&](uint32_t il) {
                            return il < hparams.n_layer() && hparams.is_recr(il);
                        };

                        if (arch == LLM_ARCH_QWEN4EXP && hparams.indexer_head_size > 0) {
                            // QSA runs on the dense-attention layers only
                            filter_idx = [&](uint32_t il) {
                                return il < hparams.n_layer() && !hparams.is_recr(il);
                            };
                        }
                    }

                    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                        // Use hybrid-iswa for hybrid models with SWA
                        res = new llama_memory_hybrid_iswa(
                            /* model             */ *this,
                            /* attn_type_k       */ params.type_k,
                            /* attn_type_v       */ params.type_v,
                            /* attn_v_trans      */ !cparams.flash_attn,
                            /* attn_swa_full     */ params.swa_full,
                            /* attn_kv_size      */ cparams.n_ctx_seq,
                            /* attn_n_ubatch     */ cparams.n_ubatch,
                            /* attn_n_pad        */ 1,
                            /* attn_offload      */ cparams.offload_kqv,
                            /* placement         */ specialized_placement,
                            /* recurrent_type_r  */ GGML_TYPE_F32,
                            /* recurrent_type_s  */ GGML_TYPE_F32,
                            /* recurrent_rs_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                            /* n_seq_max         */ cparams.n_seq_max,
                            /* n_rs_seq          */ cparams.n_rs_seq,
                            /* unified           */ cparams.kv_unified,
                            /* filter_attn       */ std::move(filter_attn),
                            /* filter_recr       */ std::move(filter_recr));
                    } else if (needs_mem_idx) {
                        // sparse attention over a per-token indexer cache, in its own memory type
                        res = new llama_memory_hybrid_idx(
                            /* model             */ *this,
                            /* attn_type_k       */ params.type_k,
                            /* attn_type_v       */ params.type_v,
                            /* attn_v_trans      */ !cparams.flash_attn,
                            /* attn_kv_size      */ cparams.n_ctx_seq,
                            /* attn_n_pad        */ 1,
                            /* attn_n_swa        */ hparams.n_swa,
                            /* attn_swa_type     */ hparams.swa_type,
                            /* recurrent_type_k  */ GGML_TYPE_F32,
                            /* recurrent_type_v  */ GGML_TYPE_F32,
                            /* recurrent_kv_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                            /* n_seq_max         */ cparams.n_seq_max,
                            /* n_rs_seq          */ cparams.n_rs_seq,
                            /* offload           */ cparams.offload_kqv,
                            /* placement         */ placement,
                            /* unified           */ cparams.kv_unified,
                            /* filter_attn       */ std::move(filter_attn),
                            /* filter_recr       */ std::move(filter_recr),
                            /* filter_idx        */ std::move(filter_idx));
                    } else {
                        res = new llama_memory_hybrid(
                            /* model             */ *this,
                            /* attn_type_k       */ params.type_k,
                            /* attn_type_v       */ params.type_v,
                            /* attn_v_trans      */ !cparams.flash_attn,
                            /* attn_kv_size      */ cparams.n_ctx_seq,
                            /* attn_n_pad        */ 1,
                            /* attn_n_swa        */ hparams.n_swa,
                            /* attn_swa_type     */ hparams.swa_type,
                            /* attn_offload      */ cparams.offload_kqv,
                            /* placement         */ placement,
                            /* recurrent_type_k  */ GGML_TYPE_F32,
                            /* recurrent_type_v  */ GGML_TYPE_F32,
                            /* recurrent_kv_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                            /* n_seq_max         */ cparams.n_seq_max,
                            /* n_rs_seq          */ cparams.n_rs_seq,
                            /* unified           */ cparams.kv_unified,
                            /* filter_attn       */ std::move(filter_attn),
                            /* filter_recr       */ std::move(filter_recr));
                    }
                } else {
                    llama_kv_cache::layer_filter_cb filter = nullptr;
                    llama_memory_i::layer_reuse_cb reuse = nullptr;
                    llama_kv_cache::layer_share_cb share = nullptr;

                    if (arch == LLM_ARCH_GEMMA3N || arch == LLM_ARCH_GEMMA4) {
                        reuse = [&](uint32_t il) {
                            GGML_ASSERT(hparams.n_layer_kv_from_start >= 2);

                            if (il >= (uint32_t)hparams.n_layer_kv_from_start) {
                                return hparams.n_layer_kv_from_start - (hparams.is_swa(il) ? 2 : 1);
                            }

                            return -1;
                        };
                    }

                    if (mtp_on_hybrid_qwen || mtp_on_hybrid_nemotron) {
                        filter = [&](uint32_t il) { return il >= hparams.n_layer(); };
                    }

                    // don't filter when n_layer_nextn is repurposed for a router layer the trunk attends
                    // or when a model is entirely n_layer_nextn layers and has no trunk
                    if (hparams.n_layer_nextn > 0 && hparams.n_layer() > 0 && hparams.router_layer < 0) {
                        if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
                            filter = [&](uint32_t il) { return il >= hparams.n_layer(); };
                        } else {
                            filter = [&](uint32_t il) { return il <  hparams.n_layer(); };
                        }
                    }

                    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                        GGML_ASSERT(hparams.is_swa_any());

                        if (arch == LLM_ARCH_GEMMA4_ASSISTANT) {
                            llama_memory_t mem_other = llama_get_memory(cparams.ctx_other);

                            share = [&](int32_t il) {
                                const llama_model * model_other = llama_get_model(cparams.ctx_other);

                                if (hparams.is_swa(il)) {
                                    return llama_model_n_layer(model_other) - 2;
                                }

                                return llama_model_n_layer(model_other) - 1;
                            };

                            res = new llama_kv_cache_iswa(
                                    *this,
                                    params.type_k,
                                    params.type_v,
                                    !cparams.flash_attn,
                                    cparams.offload_kqv,
                                    params.swa_full,
                                    cparams.kv_unified,
                                    cparams.n_ctx_seq,
                                    cparams.n_seq_max,
                                    cparams.n_ubatch,
                                    1,
                                    mem_other,
                                    filter,
                                    reuse,
                                    share,
                                    specialized_placement);
                        } else {
                            res = new llama_kv_cache_iswa(
                                    *this,
                                    params.type_k,
                                    params.type_v,
                                    !cparams.flash_attn,
                                    cparams.offload_kqv,
                                    params.swa_full,
                                    cparams.kv_unified,
                                    cparams.n_ctx_seq,
                                    cparams.n_seq_max,
                                    cparams.n_ubatch,
                                    1,
                                    nullptr,
                                    filter,
                                    reuse,
                                    share,
                                    specialized_placement);
                        }
                    } else {
                        GGML_ASSERT(!hparams.is_swa_any());

                        res = new llama_kv_cache(
                                *this,
                                hparams,
                                params.type_k,
                                params.type_v,
                                !cparams.flash_attn,
                                cparams.offload_kqv,
                                cparams.kv_unified,
                                cparams.n_ctx_seq,
                                cparams.n_seq_max,
                                1,
                                hparams.n_swa,
                                hparams.swa_type,
                                nullptr,
                                filter,
                                nullptr,
                                nullptr,
                                placement);
                    }
                }
            }
    }

    return res;
}

ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params);

    // add on pooling layer
    llm->build_pooling(cls, cls_b, cls_out, cls_out_b, cls_norm);

    // add backend sampling layers (if any)
    llm->build_sampling();

    // if the gguf model was converted with --sentence-transformers-dense-modules
    // there will be two additional dense projection layers
    // dense linear projections are applied after pooling
    // TODO: move reranking logic here and generalize
    llm->build_dense_out(dense_2_out_layers, dense_2_out_layers_b, dense_3_out_layers);

    llm->res->set_outputs(params);

    return llm->res->get_gf();
}


//
// interface implementation
//

llama_model_params llama_model_default_params() {
    llama_model_params result = {
        /*.devices                     =*/ nullptr,
        /*.tensor_buft_overrides       =*/ nullptr,
        /*.n_gpu_layers                =*/ -1,
        /*.moe_expert_cache_slots      =*/ 0,
        /*.moe_expert_cache_host_pinned_size =*/ 0,
        /*.split_mode                  =*/ LLAMA_SPLIT_MODE_LAYER,
        /*.load_mode                   =*/ LLAMA_LOAD_MODE_AUTO,
        /*.lazy_mode                   =*/ LLAMA_LAZY_MODE_AUTO,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.model_shared                =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.check_tensors               =*/ false,
        /*.use_extra_bufts             =*/ true,
        /*.no_host                     =*/ false,
        /*.no_alloc                    =*/ false,
        /*.load_mtp                    =*/ false,
        /*.moe_expert_cache_layer_ranges =*/ nullptr,
        /*.n_moe_expert_cache_layer_ranges =*/ 0,
        /*.moe_expert_cache_byte_budgets =*/ nullptr,
        /*.n_moe_expert_cache_byte_budgets =*/ 0,
    };

    return result;
}

const llama_vocab * llama_model_get_vocab(const llama_model * model) {
    return &model->vocab;
}

void llama_free_model(llama_model * model) {
    llama_model_free(model);
}

void llama_model_free(llama_model * model) {
    if (model && model->moe_expert_cache_slots() > 0) {
        for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
            ggml_backend_reg_t reg = ggml_backend_reg_get(i);
            auto log_stats_fn = (ggml_backend_moe_cache_log_and_reset_stats_t) ggml_backend_reg_get_proc_address(
                    reg, GGML_BACKEND_MOE_CACHE_LOG_AND_RESET_STATS_PROC_NAME);
            if (log_stats_fn != nullptr) {
                log_stats_fn();
            }
        }
    }
    delete model;
}

int32_t llama_model_n_ctx_train(const llama_model * model) {
    return model->hparams.n_ctx_train;
}

int32_t llama_model_n_embd(const llama_model * model) {
    return model->hparams.n_embd;
}

int32_t llama_model_n_embd_inp(const llama_model * model) {
    return model->hparams.n_embd_inp();
}

int32_t llama_model_n_embd_out(const llama_model * model) {
    return model->hparams.n_embd_out();
}

int32_t llama_model_n_layer(const llama_model * model) {
    return model->hparams.n_layer();
}

int32_t llama_model_n_layer_nextn(const llama_model * model) {
    return model->hparams.n_layer_nextn;
}

int32_t llama_model_dflash_selector_top_k(const llama_model * model) {
    return model->hparams.dflash_selector_top_k;
}

int32_t llama_model_n_head(const llama_model * model) {
    return model->hparams.n_head();
}

int32_t llama_model_n_head_kv(const llama_model * model) {
    return model->hparams.n_head_kv();
}

int32_t llama_model_n_swa(const llama_model * model) {
    // dsv4 kv-cache has SWA but it cannot be used as a rollback because of
    // other compression ratios, so we return 0 here
    if (model->arch == LLM_ARCH_DEEPSEEK4) {
        return 0;
    }
    return model->hparams.n_swa;
}


uint32_t llama_model_n_cls_out(const struct llama_model * model) {
    return model->hparams.n_cls_out;
}

const char * llama_model_cls_label(const struct llama_model * model, uint32_t i) {
    if (i < model->classifier_labels.size()) {
        return model->classifier_labels[i].c_str();
    }

    return nullptr;
}

// deprecated
int32_t llama_n_ctx_train(const llama_model * model) {
    return llama_model_n_ctx_train(model);
}

// deprecated
int32_t llama_n_embd(const llama_model * model) {
    return llama_model_n_embd(model);
}

// deprecated
int32_t llama_n_layer(const llama_model * model) {
    return llama_model_n_layer(model);
}

// deprecated
int32_t llama_n_head(const llama_model * model) {
    return llama_model_n_head(model);
}

llama_rope_type llama_model_rope_type(const llama_model * model) {
    switch (model->arch) {
        // these models do not use RoPE
        case LLM_ARCH_CLIP:
        case LLM_ARCH_GPT2:
        case LLM_ARCH_GPTJ:
        case LLM_ARCH_MPT:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_BLOOM:
        case LLM_ARCH_MAMBA:
        case LLM_ARCH_MAMBA2:
        case LLM_ARCH_JAMBA:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_T5:
        case LLM_ARCH_T5ENCODER:
        case LLM_ARCH_JAIS:
        case LLM_ARCH_RWKV6:
        case LLM_ARCH_RWKV6QWEN2:
        case LLM_ARCH_RWKV7:
        case LLM_ARCH_ARWKV7:
        case LLM_ARCH_WAVTOKENIZER_DEC:
        case LLM_ARCH_NEMOTRON_H:
        case LLM_ARCH_NEMOTRON_H_MOE:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
            return LLAMA_ROPE_TYPE_NONE;

        // use what we call a normal RoPE, operating on pairs of consecutive head values
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_DECI:
        case LLM_ARCH_BAICHUAN:
        case LLM_ARCH_STARCODER:
        case LLM_ARCH_INTERNLM2:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_XVERSE:
        case LLM_ARCH_COMMAND_R:
        case LLM_ARCH_COHERE2:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_OLMO:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK2OCR:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_MUSE_GLIMMER:
        case LLM_ARCH_PLM:
        case LLM_ARCH_CHATGLM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_GRANITE_HYBRID:
        case LLM_ARCH_GRANITE_SWITCH:
        case LLM_ARCH_GRANITE_SWA:
        case LLM_ARCH_CHAMELEON:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_SMOLLM3:
        case LLM_ARCH_ARCEE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_EAGLE3:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_LLAMA_EMBED:
        case LLM_ARCH_MAINCODER:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_NANBEIGE:
        case LLM_ARCH_POCKETTTS:
        // HY_V4 rotates consecutive pairs, matching the reference implementation
        case LLM_ARCH_HY_V4:
            return LLAMA_ROPE_TYPE_NORM;

        // the pairs of head values are offset by n_rot/2
        case LLM_ARCH_FALCON:
        case LLM_ARCH_FALCON_H1:
        case LLM_ARCH_GROK:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V3:
        case LLM_ARCH_MODERN_BERT:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_EUROBERT:
        case LLM_ARCH_STABLELM:
        case LLM_ARCH_BITNET:
        case LLM_ARCH_QWEN:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_DREAM:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_RND1:
        case LLM_ARCH_OLMO2:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_PHI2:
        case LLM_ARCH_PHI3:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_PLAMO:
        case LLM_ARCH_PLAMO2:
        case LLM_ARCH_PLAMO3:
        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA2:
        case LLM_ARCH_GEMMA3:
        case LLM_ARCH_GEMMA3N:
        case LLM_ARCH_GEMMA4:
        case LLM_ARCH_GEMMA4_ASSISTANT:
        case LLM_ARCH_GEMMA_EMBEDDING:
        case LLM_ARCH_STARCODER2:
        case LLM_ARCH_OPENELM:
        case LLM_ARCH_GPTNEOX:
        case LLM_ARCH_CODESHELL:
        case LLM_ARCH_ORION:
        case LLM_ARCH_NEMOTRON:
        case LLM_ARCH_EXAONE:
        case LLM_ARCH_EXAONE4:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_MINICPM3:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_JAIS2:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_HUNYUAN_DENSE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_LFM2:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_SEED_OSS:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_APERTUS:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_COGVLM:
        case LLM_ARCH_PANGU_EMBED:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_LAGUNA:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_SPARK2_5:
        case LLM_ARCH_TALKIE:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_MAPLE:
        case LLM_ARCH_HRM_TEXT:
            return LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_DFLASH:
            // drafts for M-RoPE targets carry rope sections and follow the target's temporal dim
            if (const auto & s = model->hparams.rope_sections; s[0] || s[1] || s[2] || s[3]) {
                return LLAMA_ROPE_TYPE_MROPE;
            }
            // DSV4 DSpark drafters use DeepSeek-V4's normal RoPE; legacy DFlash backbones are NeoX
            return model->hparams.dsv4_hc_mult > 0 ? LLAMA_ROPE_TYPE_NORM : LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_QWEN2VL:
        case LLM_ARCH_PADDLEOCR:
            return LLAMA_ROPE_TYPE_MROPE;
        case LLM_ARCH_QWEN3VL:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_QWEN3TTS:
            return LLAMA_ROPE_TYPE_IMROPE;

        case LLM_ARCH_GLM4:
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NORM;
        case LLM_ARCH_GLM4_MOE:
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_HUNYUAN_VL:
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NEOX;

        // all model arches should be listed explicitly here
        case LLM_ARCH_UNKNOWN:
            GGML_ABORT("unknown architecture");
    }

    return LLAMA_ROPE_TYPE_NONE;
}

float llama_model_rope_freq_scale_train(const llama_model * model) {
    return model->hparams.rope_freq_scale_train;
}

int32_t llama_model_meta_val_str(const llama_model * model, const char * key, char * buf, size_t buf_size) {
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_meta_count(const llama_model * model) {
    return (int)model->gguf_kv.size();
}

const char * llama_model_meta_key_str(llama_model_meta_key key) {
    switch (key) {
        case LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE:        return "general.sampling.sequence";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_K:           return "general.sampling.top_k";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_P:           return "general.sampling.top_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIN_P:           return "general.sampling.min_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY: return "general.sampling.xtc_probability";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD:   return "general.sampling.xtc_threshold";
        case LLAMA_MODEL_META_KEY_SAMPLING_TEMP:            return "general.sampling.temp";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N:  return "general.sampling.penalty_last_n";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT:  return "general.sampling.penalty_repeat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT:        return "general.sampling.mirostat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU:    return "general.sampling.mirostat_tau";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA:    return "general.sampling.mirostat_eta";
        default:                                            return nullptr;
    }
}

int32_t llama_model_meta_key_by_index(const llama_model * model, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_model_meta_val_str_by_index(const llama_model * model, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_desc(const llama_model * model, char * buf, size_t buf_size) {
    return snprintf(buf, buf_size, "%s", model->desc().c_str());
}

llama_ftype llama_model_ftype(const llama_model * model) {
    return model->ftype();
}

uint64_t llama_model_size(const llama_model * model) {
    return model->size();
}

const char * llama_model_chat_template(const llama_model * model, const char * name) {
    const auto key = name ? LLM_KV(model->arch, name)(LLM_KV_TOKENIZER_CHAT_TEMPLATE)
        : LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        // one-off fix for very popular models (so we are not flooded with issues)
        // do not extend this list unless absolutely necessary
        // Mistral-Small-2503 does not have built-in chat template
        llama_vocab_pre_type pre_type = model->vocab.get_pre_type();
        if (!name && pre_type == LLAMA_VOCAB_PRE_TYPE_TEKKEN && model->layers.size() == 40) {
            return "mistral-v7-tekken";
        }

        return nullptr;
    }

    return it->second.c_str();
}

uint64_t llama_model_n_params(const llama_model * model) {
    return model->n_elements();
}

bool llama_model_has_encoder(const llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5:
        case LLM_ARCH_T5ENCODER:
        case LLM_ARCH_EAGLE3:
        case LLM_ARCH_DFLASH:    return true;
        default:                 return false;
    }
}

bool llama_model_has_decoder(const llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5ENCODER: return false;
        default:                 return true;
    }
}

llama_token llama_model_decoder_start_token(const llama_model * model) {
    return model->hparams.dec_start_token_id;
}

bool llama_model_is_recurrent(const llama_model * model) {
    return llm_arch_is_recurrent(model->arch);
}

bool llama_model_is_hybrid(const llama_model * model) {
    return llm_arch_is_hybrid(model->arch);
}

bool llama_model_is_diffusion(const llama_model * model) {
    return llm_arch_is_diffusion(model->arch);
}

const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model) {
    return model->tensors_by_name;
}

bool llama_internal_model_no_alloc(const llama_model * model) {
    return model != nullptr && model->is_no_alloc();
}

int32_t llama_model_n_expert(const struct llama_model * model) {
    return model->hparams.n_expert;
}

int32_t llama_model_n_devices(const struct llama_model * model) {
    return (int32_t)model->devices.size();
}

ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i) {
    if (i < 0 || i >= (int)model->devices.size()) {
        return nullptr;
    }
    return model->devices[i].dev;
}

//
// llama_model_base
//

llama_model_base::llama_model_base(const struct llama_model_params & params) : llama_model(params), model(this), tn(model->arch),
    TENSOR_DUPLICATED     (llama_model_loader::TENSOR_DUPLICATED),
    TENSOR_NOT_REQUIRED   (llama_model_loader::TENSOR_NOT_REQUIRED),
    TENSOR_SKIP           (llama_model_loader::TENSOR_SKIP),
    TENSOR_SKIP_IF_VIRTUAL(llama_model_loader::TENSOR_SKIP_IF_VIRTUAL),
    TENSOR_ALLOW_RESHAPE  (llama_model_loader::TENSOR_ALLOW_RESHAPE),
    TENSOR_READ_LAZY      (llama_model_loader::TENSOR_READ_LAZY) {}

ggml_tensor * llama_model_base::create_tensor(const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    GGML_ASSERT(ml != nullptr);
    return create_tensor(*ml, tn, ne, flags);
}

void llama_model_base::create_tensor_gate_up_exps(llama_layer & layer, int bid, int64_t n_embd_, int64_t n_ff_, int64_t n_expert_, int flags) {
    if (flags & TENSOR_SKIP) {
        const int skip = TENSOR_NOT_REQUIRED | TENSOR_SKIP;

        create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", bid), {n_embd_, n_ff_ * 2, n_expert_}, skip | TENSOR_SKIP_IF_VIRTUAL);
        create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS,    "weight", bid), {n_embd_, n_ff_,     n_expert_}, skip);
        create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,      "weight", bid), {n_embd_, n_ff_,     n_expert_}, skip);
        return;
    }

    layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", bid), {n_embd_, n_ff_ * 2, n_expert_}, TENSOR_NOT_REQUIRED);
    if (layer.ffn_gate_up_exps == nullptr) {
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
    }
}

void llama_model_base::create_tensor_qkv(llama_layer & layer, int bid,
        int64_t n_embd_, int64_t n_embd_q_, int64_t n_embd_k_, int64_t n_embd_v_,
        int flags) {
    const int64_t n_embd_qkv = n_embd_q_ + n_embd_k_ + n_embd_v_;

    if (flags & TENSOR_SKIP) {
        const int skip = TENSOR_NOT_REQUIRED | TENSOR_SKIP;

        create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", bid), {n_embd_, n_embd_qkv}, skip | TENSOR_SKIP_IF_VIRTUAL);
        create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias",   bid), {n_embd_qkv},          skip | TENSOR_SKIP_IF_VIRTUAL);
        create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", bid), {n_embd_, n_embd_q_},  skip);
        create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", bid), {n_embd_, n_embd_k_},  skip);
        create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", bid), {n_embd_, n_embd_v_},  skip);
        create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias",   bid), {n_embd_q_},           skip);
        create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias",   bid), {n_embd_k_},           skip);
        create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias",   bid), {n_embd_v_},           skip);
        return;
    }

    layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", bid), {n_embd_, n_embd_qkv}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
    if (layer.wqkv) {
        layer.wqkv_b = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", bid), {n_embd_qkv}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
        // Fused weights may coexist with separate Q/K/V biases in legacy or custom GGUFs.
        if (!layer.wqkv_b) {
            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", bid), {n_embd_q_}, TENSOR_NOT_REQUIRED);
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", bid), {n_embd_k_}, TENSOR_NOT_REQUIRED);
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", bid), {n_embd_v_}, TENSOR_NOT_REQUIRED);
        }
    } else {
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", bid), {n_embd_, n_embd_q_}, flags);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", bid), {n_embd_, n_embd_k_}, flags);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", bid), {n_embd_, n_embd_v_}, flags);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", bid), {n_embd_q_}, TENSOR_NOT_REQUIRED);
        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", bid), {n_embd_k_}, TENSOR_NOT_REQUIRED);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", bid), {n_embd_v_}, TENSOR_NOT_REQUIRED);
    }
}

void llama_model_base::load_swa_pattern(llama_model_loader & ml, uint32_t n_pattern, bool dense_first) {
    if (ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, false)) {
        return;
    }

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern, false);
    hparams.set_swa_pattern(n_pattern, dense_first);
}

const int32_t * llama_model_target_layer_ids(const struct llama_model * model) {
    const auto & v = model->target_layer_ids;
    return v.empty() ? nullptr : v.data();
}

uint32_t llama_model_target_layer_ids_n(const struct llama_model * model) {
    return (uint32_t) model->target_layer_ids.size();
}

uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out) {
    if (model->vocab.n_tokens() == 0 || model->tok_embd == nullptr) {
        return 0;
    }

    const ggml_tensor * tensor = model->tok_embd;
    const size_t nelements = ggml_nelements(tensor);
    GGML_ASSERT(nelements <= UINT32_MAX); // for the return type

    if (out == nullptr) {
        return (uint32_t) nelements;
    }

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, out, 0, nelements * sizeof(float));
        return (uint32_t) nelements;
    }

    std::vector<uint8_t> buf(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, buf.data(), 0, buf.size());

    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) buf.data(), out, nelements);
    } else if (tensor->type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) buf.data(), out, nelements);
    } else if (ggml_is_quantized(tensor->type) && traits->to_float != nullptr) {
        traits->to_float(buf.data(), out, nelements);
    } else {
        GGML_ABORT("unsupported tensor type for dequantization: %s", ggml_type_name(tensor->type));
    }

    return (uint32_t) nelements;
}
