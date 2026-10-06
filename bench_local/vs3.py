"""Interleaved: Photon + any number of (bench_binary, model) pairs. Usage: vs3.py ROUNDS THREADS bin:model [bin:model ...]"""
import os, subprocess, statistics, sys, time
import moondream as md
WAV, DUR = "/Users/nabilhossain/StudioProjects/parakeet.cpp/models/clean.wav", 24.13
rounds, threads, pairs = int(sys.argv[1]), sys.argv[2], [p.split(":") for p in sys.argv[3:]]
res = {"photon": []} | {f"{os.path.basename(b)}:{os.path.basename(m)}": [] for b, m in pairs}
kw = {} if threads == "default" else {"cpu_threads": int(threads)}
with md.photon("moondream/parakeet-redux", device="cpu", **kw) as s:
    s.transcribe(audio=WAV)
    for _ in range(rounds):
        t = time.time(); s.transcribe(audio=WAV); res["photon"].append(DUR / (time.time() - t))
        for b, m in pairs:
            env = dict(os.environ, DYLD_LIBRARY_PATH=open(b + ".libpath").read().strip())
            out = subprocess.run([b, m, WAV, "8" if threads == "default" else threads, "2"], env=env, capture_output=True, text=True).stdout
            res[f"{os.path.basename(b)}:{os.path.basename(m)}"] += [float(x) for x in out.split()]
for k, v in res.items():
    print(f"  {k:<44} median {statistics.median(v):6.1f}x  [{min(v):.1f} .. {max(v):.1f}]")
