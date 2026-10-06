// Times encoder+decoder on one clip: N timed runs after one warmup; prints each run's RTF.
#include "model.hpp"
#include "audio_io.hpp"
#include "ggml_graph.hpp"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif
// current physical footprint in MB (what iOS jetsam / Android LMK look at)
static double rss_now_mb() {
#ifdef __APPLE__
    task_vm_info_data_t vi; mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &vi, &n) != KERN_SUCCESS) return -1;
    return vi.phys_footprint / 1048576.0;
#else
    long pages = 0, rss = 0; FILE* f = fopen("/proc/self/statm", "r");
    if (!f || fscanf(f, "%ld %ld", &pages, &rss) != 2) { if (f) fclose(f); return -1; }
    fclose(f); return rss * 4096.0 / 1048576.0;
#endif
}
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: bench <model.gguf> <wav> [threads] [runs] [print_text]\n"); return 1; }
    const int threads = argc > 3 ? atoi(argv[3]) : 8;
    const int runs    = argc > 4 ? atoi(argv[4]) : 7;
    pk::set_num_threads(threads);
    pk::Audio a; pk::load_audio_16k_mono(argv[2], a);
    auto m = pk::Model::load(argv[1]);
    std::string t = m->transcribe_16k(a.samples, pk::Decoder::kDefault, "");
    const double dur = a.samples.size() / 16000.0;
    for (int i = 0; i < runs; ++i) {
        auto s = std::chrono::steady_clock::now();
        t = m->transcribe_16k(a.samples, pk::Decoder::kDefault, "");
        double e = std::chrono::duration<double>(std::chrono::steady_clock::now() - s).count();
        printf("%.2f\n", dur / e);
    }
    if (argc > 5) printf("TEXT %s\n", t.c_str());
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    printf("PEAK_RSS_MB %.1f\n", ru.ru_maxrss / 1048576.0);
#else
    printf("PEAK_RSS_MB %.1f\n", ru.ru_maxrss / 1024.0);
#endif
    printf("RSS_NOW_MB %.1f\n", rss_now_mb());
    pk::shutdown_backend();
    return 0;
}
