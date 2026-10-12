#include "common.cuh"

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream);
void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_row_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * mul, ggml_tensor * sum, ggml_tensor * const * post, const int * kind);
