// NEON quantize_row_q8_K (CPU type traits) vs ggml reference, byte-exact.
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
void quantize_row_q8_K_ref(const float * x, void * y, int64_t k);
int main(void) {
    ggml_cpu_init();
    const int nb = 4096, k = nb * 256;
    float * x = malloc(sizeof(float) * k);
    srand(1);
    for (int i = 0; i < k; ++i) x[i] = ((rand() / (float) RAND_MAX) - 0.5f) * powf(10.f, (rand() % 7) - 3);
    for (int i = 0; i < 256; ++i) x[256 + i] = 0.0f;                 // all-zero block
    x[512 + 3] = 2.0f; x[512 + 9] = -2.0f;                            // tie on |max|, first wins
    for (int i = 768; i < 1024; ++i) x[i] = (i % 7) * 0.5f - 1.5f;    // many exact .5 ties after scaling
    printf("ref=%p cpu=%p\n", (void*) ggml_get_type_traits(GGML_TYPE_Q8_K)->from_float_ref, (void*) ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float); fflush(stdout);
    const size_t bs = ggml_row_size(GGML_TYPE_Q8_K, k);
    void * a = malloc(bs), * b = malloc(bs);
    quantize_row_q8_K_ref(x, a, k);
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x, b, k);
    // compare per block, skipping bsums of the zero block (ref leaves them unset)
    const size_t blk = bs / nb; int bad = 0;
    for (int i = 0; i < nb; ++i) {
        const size_t n = (i == 1) ? 4 + 256 : blk;
        if (memcmp((char *) a + i * blk, (char *) b + i * blk, n)) { if (bad < 3) printf("block %d differs\n", i); bad++; }
    }
    printf("%d / %d blocks differ\n", bad, nb);
    return bad != 0;
}
