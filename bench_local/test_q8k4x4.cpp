// NEON ggml_quantize_mat_q8_K_4x4 vs the generic version, byte-exact.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
extern "C" void ggml_quantize_mat_q8_K_4x4(const float * x, void * vy, int64_t k);
extern "C" void ggml_quantize_mat_q8_K_4x4_generic(const float * x, void * vy, int64_t k);
int main() {
    const int64_t k = 4096; const size_t blk = 4 * 4 + 256 * 4 + 64 * 2, nb = k / 256;
    std::vector<float> x(4 * k);
    srand(3);
    for (auto & v : x) v = ((rand() / (float) RAND_MAX) - 0.5f) * powf(10.f, (rand() % 7) - 3);
    for (int i = 0; i < 256; ++i) x[k + i] = 0.0f;               // all-zero block in row 1
    x[2 * k + 5] = 3.0f; x[2 * k + 11] = -3.0f;                    // |max| tie, first wins
    std::vector<uint8_t> a(blk * nb), b(blk * nb);
    ggml_quantize_mat_q8_K_4x4_generic(x.data(), a.data(), k);
    ggml_quantize_mat_q8_K_4x4(x.data(), b.data(), k);
    int bad = 0;
    for (size_t i = 0; i < nb; ++i) if (memcmp(&a[i * blk], &b[i * blk], blk)) { if (bad < 3) printf("block %zu differs\n", i); bad++; }
    printf("%d / %zu blocks differ\n", bad, nb);
    return bad != 0;
}
