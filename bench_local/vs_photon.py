"""Interleaved Photon-vs-ggml CPU benchmark: each round runs Photon once and each ggml model once,
so background load hits all of them equally. Reports median RTF per runtime."""
import os, subprocess, statistics, sys, time
import moondream as md

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAV, DUR = os.path.join(ROOT, "models/clean.wav"), 24.13
models = sys.argv[2:] or ["models/redux-tq2_0.gguf"]
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 12
libs = ":".join(sorted({os.path.dirname(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(ROOT, "build-cpu"))
                        for f in fs if f.startswith("libggml") and f.endswith(".dylib")}))
env = dict(os.environ, DYLD_LIBRARY_PATH=libs)
res = {"photon": []} | {m: [] for m in models}
with md.photon("moondream/parakeet-redux", device="cpu", **({"cpu_threads": int(os.environ["PHOTON_THREADS"])} if os.environ.get("PHOTON_THREADS") else {})) as s:
    s.transcribe(audio=WAV)
    for r in range(rounds):
        t = time.time(); s.transcribe(audio=WAV); res["photon"].append(DUR / (time.time() - t))
        for m in models:
            BIN = os.environ.get("BIN", "bench_local/bench"); env["DYLD_LIBRARY_PATH"] = open(os.path.join(ROOT, BIN + ".libpath")).read().strip() if os.path.exists(os.path.join(ROOT, BIN + ".libpath")) else env["DYLD_LIBRARY_PATH"]
            out = subprocess.run([os.path.join(ROOT, BIN), os.path.join(ROOT, m), WAV,
                                  os.environ.get("THREADS", "8"), "2"], env=env, capture_output=True, text=True).stdout
            res[m] += [float(x) for x in out.split()]
for k, v in res.items():
    print(f"{k:<30} median {statistics.median(v):6.1f}x   [{min(v):.1f} .. {max(v):.1f}]  n={len(v)}")
p = statistics.median(res["photon"])
for m in models:
    print(f"gap {m}: Photon is {p / statistics.median(res[m]):.2f}x faster")
