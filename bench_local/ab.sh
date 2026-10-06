#!/bin/bash
# usage: ab.sh rounds model1.gguf model2.gguf ...   -> median RTF per model, rounds interleaved
cd "$(dirname "$0")/.."
L=$(find build-cpu -name "libggml*.dylib" -exec dirname {} \; | sort -u | tr '\n' ':')
R=$1; shift
for m in "$@"; do : > /tmp/ab_$(basename $m).txt; done
for r in $(seq 1 $R); do
  for m in "$@"; do DYLD_LIBRARY_PATH=$L bench_local/bench $m models/clean.wav ${THREADS:-8} 3 >> /tmp/ab_$(basename $m).txt; done
done
for m in "$@"; do printf "%-28s median %6.1fx   (n=%d)\n" $(basename $m) $(sort -n /tmp/ab_$(basename $m).txt | awk '{a[NR]=$1} END{print (NR%2? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}') $(wc -l < /tmp/ab_$(basename $m).txt); done
