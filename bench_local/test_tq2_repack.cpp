// TQ2_0 matmul: normal CPU buffer (vec_dot path) vs repack buffer (new i8mm gemm/gemv path).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<float> run(ggml_backend_t be, ggml_backend_buffer_type_t wbuft, const std::vector<uint8_t> & wq,
                              const std::vector<float> & xin, int K, int R, int T, int threads) {
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * cw = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(cw, GGML_TYPE_TQ2_0, K, R);
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, wbuft);
    ggml_backend_tensor_set(W, wq.data(), 0, wq.size());

    ggml_context * cx = ggml_init(ip);
    ggml_tensor * X = ggml_new_tensor_2d(cx, GGML_TYPE_F32, K, T);
    ggml_tensor * Y = ggml_mul_mat(cx, W, X);
    ggml_cgraph * gf = ggml_new_graph(cx);
    ggml_build_forward_expand(gf, Y);
    ggml_backend_buffer_t bx = ggml_backend_alloc_ctx_tensors(cx, be);
    ggml_backend_tensor_set(X, xin.data(), 0, xin.size() * sizeof(float));
    ggml_backend_cpu_set_n_threads(be, threads);
    ggml_backend_graph_compute(be, gf);
    std::vector<float> out((size_t) R * T);
    ggml_backend_tensor_get(Y, out.data(), 0, out.size() * sizeof(float));
    ggml_backend_buffer_free(bx); ggml_backend_buffer_free(bw); ggml_free(cx); ggml_free(cw);
    return out;
}

int main() {
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_dev_t dev = ggml_backend_get_device(be);
    auto get_extra = (ggml_backend_buffer_type_t * (*)(ggml_backend_dev_t))
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t repack = nullptr;
    for (auto * b = get_extra(dev); b && *b; ++b) {
        printf("extra buft: %s\n", ggml_backend_buft_name(*b));
        if (!repack) repack = *b;
    }
    if (!repack) { printf("no repack buffer type\n"); return 1; }

    int fails = 0;
    for (int T : {1, 3, 4, 7, 16, 19, 20, 33, 302}) {
        const int K = 1024, R = 64;
        srand(T);
        std::vector<float> wf((size_t) K * R), xin((size_t) K * T);
        for (auto & v : wf)  v = (float) ((rand() % 3) - 1) * (0.5f + (rand() % 100) / 100.f);
        for (auto & v : xin) v = ((rand() / (float) RAND_MAX) - 0.5f) * 4.f;
        std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_TQ2_0, K) * R);
        ggml_quantize_chunk(GGML_TYPE_TQ2_0, wf.data(), wq.data(), 0, R, K, nullptr);

        auto ref = run(be, ggml_backend_cpu_buffer_type(), wq, xin, K, R, T, 4);
        auto got = run(be, repack, wq, xin, K, R, T, 4);
        double maxrel = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double d = std::fabs(ref[i] - got[i]) / std::max(1e-3, (double) std::fabs(ref[i]));
            maxrel = std::max(maxrel, d);
        }
        printf("T=%-4d max rel err %.2e %s\n", T, maxrel, maxrel < 1e-4 ? "OK" : "MISMATCH");
        fails += maxrel >= 1e-4;
    }
    ggml_backend_free(be);
    return fails;
}
