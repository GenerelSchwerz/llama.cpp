// Adapted from MIT-licensed expert primitives; see moe-reference-LICENSE.
// See moe-reference-LICENSE and the source inventory in LLAMA-INTEGRATION-DELTA.md.
#include "moe-reference.h"
#include <immintrin.h>
#include <cstring>
namespace ggml_moe_reference {
inline __m512i unpack64_q2_0(const uint8_t* codes) {
    const __m128i packed = _mm_loadu_si128((const __m128i*) codes);          // 8 u16 = 64 codes
    const __m512i lanes = _mm512_cvtepu16_epi64(packed);                    // 8 qwords, 8 codes each
    const __m512i ctrl = _mm512_set1_epi64((long long) 0x0E0C0A0806040200ULL);
    return _mm512_and_si512(_mm512_multishift_epi64_epi8(ctrl, lanes), _mm512_set1_epi8(3));
}

void act_quant_q8_1(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    // Plan v0.3 P6: AVX-512, the same operations per element as the scalar loop below (max of |x|, one multiply,
    // +-0.5 away from zero, truncation, clamp), so the result is bitwise the scalar one.  The scalar loop took
    // ~22 us per 2560 values - 3.2 ms of every speculative round.
    {
        const __m512 half = _mm512_set1_ps(0.5f), mhalf = _mm512_set1_ps(-0.5f), zero = _mm512_setzero_ps();
        const __m512i lo = _mm512_set1_epi32(-127), hi = _mm512_set1_epi32(127);
        const __m512 absmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
        for (int k = 0; k < a.nchunks; ++k) {
            const float* xb = x + k * QKA;
            const __m512 x0 = _mm512_loadu_ps(xb), x1 = _mm512_loadu_ps(xb + 16);
            const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_and_ps(x0, absmask), _mm512_and_ps(x1, absmask)));
            const float s = amax > 0.f ? amax / 127.f : 0.f;
            const float inv = s > 0.f ? 1.f / s : 0.f;
            const __m512 vinv = _mm512_set1_ps(inv);
            const __m512 t0 = _mm512_mul_ps(x0, vinv), t1 = _mm512_mul_ps(x1, vinv);
            const __m512 r0 = _mm512_add_ps(t0, _mm512_mask_blend_ps(_mm512_cmp_ps_mask(t0, zero, _CMP_GE_OQ), mhalf, half));
            const __m512 r1 = _mm512_add_ps(t1, _mm512_mask_blend_ps(_mm512_cmp_ps_mask(t1, zero, _CMP_GE_OQ), mhalf, half));
            __m512i v0 = _mm512_cvttps_epi32(r0), v1 = _mm512_cvttps_epi32(r1);
            v0 = _mm512_min_epi32(_mm512_max_epi32(v0, lo), hi);
            v1 = _mm512_min_epi32(_mm512_max_epi32(v1, lo), hi);
            _mm_storeu_si128((__m128i*) (a.q + k * QKA), _mm512_cvtepi32_epi8(v0));
            _mm_storeu_si128((__m128i*) (a.q + k * QKA + 16), _mm512_cvtepi32_epi8(v1));
            const int32_t sum = _mm512_reduce_add_epi32(_mm512_add_epi32(v0, v1));
            a.scale[k] = s;
            a.sum[k] = sum;
            a.hx[k] = s * (float) sum;
        }
        return;
    }
}
namespace {
template<int NT>
inline void q2g_row_multi(const uint8_t* row, const ActQ* const* a, int nblocks, float* res) {
    __m512 acc[NT], corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm512_setzero_ps(); corr[t] = _mm512_setzero_ps(); }
    const __m512i base = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    const __m512i dup = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    for (int b0 = 0; b0 < nblocks; b0 += 8) {
        const int nb = nblocks - b0 < 8 ? nblocks - b0 : 8;
        const __mmask16 m16 = nb >= 8 ? (__mmask16) 0xFFFF : (__mmask16) ((1u << (2 * nb)) - 1u);
        alignas(16) uint16_t sc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < nb; ++i) std::memcpy(&sc[i], row + (size_t) (b0 + i) * 18, 2);
        const __m512 d8 = _mm512_castps256_ps512(_mm256_cvtph_ps(_mm_load_si128((const __m128i*) sc)));
        const __m512 dd = _mm512_maskz_mov_ps(m16, _mm512_permutexvar_ps(dup, d8));
        __m512 p[NT];
        for (int t = 0; t < NT; ++t) {
            p[t] = _mm512_mul_ps(dd, _mm512_maskz_loadu_ps(m16, a[t]->scale + 2 * b0));
            corr[t] = _mm512_fmadd_ps(dd, _mm512_maskz_loadu_ps(m16, a[t]->hx + 2 * b0), corr[t]);
        }
        for (int i = 0; i < nb; ++i) {
            const int b = b0 + i;
            const __m512i w = unpack64_q2_0(row + (size_t) b * 18 + 2);
            const __m512i idx = _mm512_add_epi32(base, _mm512_set1_epi32(2 * i));
            for (int t = 0; t < NT; ++t) {
                const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w,
                                                        _mm512_load_si512((const void*) (a[t]->q + b * QK)));
                acc[t] = _mm512_fmadd_ps(_mm512_permutexvar_ps(idx, p[t]), _mm512_cvtepi32_ps(dot), acc[t]);
            }
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(acc[t]) - _mm512_reduce_add_ps(corr[t]);
}
template<int NT>
void q2g_rows(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r0, int r1) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        q2g_row_multi<NT>(w + (size_t) r * row_bytes, a, nblocks, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}
}  // namespace

void q2_0_gguf_rows_multi(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                          float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: q2g_rows<1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 2: q2g_rows<2>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 3: q2g_rows<3>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: q2g_rows<4>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 5: q2g_rows<5>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 6: q2g_rows<6>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 7: q2g_rows<7>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default: q2g_rows<8>(w, row_bytes, nblocks, a, out, r0, r1); break;
    }
}

}
