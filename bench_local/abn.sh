#!/bin/bash
# abn.sh <threads> <rounds> <model> <bin1> [bin2 ...] : interleaved medians
cd "$(dirname "$0")/.."
T=$1; R=$2; M=$3; shift 3
for b in "$@"; do : > /tmp/abn_$b.txt; done
for r in $(seq 1 $R); do for b in "$@"; do
  DYLD_LIBRARY_PATH=$(cat bench_local/$b.libpath) bench_local/$b $M models/clean.wav $T 2 >> /tmp/abn_$b.txt; done; done
for b in "$@"; do printf "  %-16s %6.2fx\n" $b $(sort -n /tmp/abn_$b.txt | awk '{a[NR]=$1} END{print (a[int((NR+1)/2)]+a[int(NR/2)+1])/2}'); done
