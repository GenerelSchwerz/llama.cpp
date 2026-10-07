// Adapted from MIT-licensed expert primitives; see moe-reference-LICENSE.
// See moe-reference-LICENSE and the source inventory in LLAMA-INTEGRATION-DELTA.md.
#pragma once
#include <cstddef>
#include <cstdint>
namespace ggml_moe_reference {
constexpr int QKA = 32;
constexpr int QK = 64;
struct ActQ { int8_t * q; float * scale; int32_t * sum; float * hx; int nchunks; };
bool iq256_supported(int type) noexcept;
bool iq512_supported(int type) noexcept;
void iq256_gu_rows(int type, const uint8_t * gate, const uint8_t * up, size_t gate_row, size_t up_row, int n, const void * const * act, int nt, float * const * ff, int r0, int r1);
void iq512_gu_rows(int type, const uint8_t * gate, const uint8_t * up, size_t gate_row, size_t up_row, int n, const void * const * act, int nt, float * const * ff, int r0, int r1);
void iq4nl256_down_rows(const uint8_t * w, size_t row_bytes, int n, const void * const * hq, int nt, float * const * out, int r0, int r1);
void act_quant_q8_1_avx2(const float * x, int n, ActQ & a);
void act_quant_q8_1(const float * x, int n, ActQ & a);
void q2_0_gguf_rows_multi_avx2(const uint8_t * w, size_t row_bytes, int nblocks, const ActQ * const * a, int nt, float * const * out, int r0, int r1);
void q2_0_gguf_rows_multi(const uint8_t * w, size_t row_bytes, int nblocks, const ActQ * const * a, int nt, float * const * out, int r0, int r1);
}
