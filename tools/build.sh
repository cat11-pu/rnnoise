#!/bin/sh
# Builds the RNNoise library plus the sample-rate-aware tools without
# requiring autotools. Run from the repository root.
set -e

cd "$(dirname "$0")/.."
mkdir -p build

RNNOISE_SRC="src/denoise.c src/rnn.c src/pitch.c src/kiss_fft.c \
src/celt_lpc.c src/nnet.c src/nnet_default.c src/parse_lpcnet_weights.c \
src/rnnoise_tables.c src/rnnoise_data.c"

CFLAGS="-O2 -Iinclude -Isrc -Wall"

for src in $RNNOISE_SRC; do
  obj="build/$(basename "$src" .c).o"
  gcc $CFLAGS -c "$src" -o "$obj"
done

ar rcs build/librnnoise.a build/*.o

gcc $CFLAGS tools/resample.c tools/rnnoise_sr.c \
  -Lbuild -lrnnoise -lm -o build/rnnoise_sr
gcc $CFLAGS tools/rnnoise_naive.c \
  -Lbuild -lrnnoise -lm -o build/rnnoise_naive
gcc $CFLAGS tools/resample.c tools/resample_pcm.c \
  -lm -o build/resample_pcm
gcc $CFLAGS eval/eval_pcm.c -lm -o build/eval_pcm

echo "built build/rnnoise_sr build/rnnoise_naive build/resample_pcm build/eval_pcm"
