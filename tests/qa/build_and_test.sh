#!/bin/sh
# Reproducible RNNoise QA: build the library from the current source tree,
# generate the 500-sample noisy-speech set, run the 500-sample QA pass and
# the 3-hour stability run. All outputs land in tests/qa/results/.
set -e

cd "$(dirname "$0")"
ROOT=../..
BUILD=build
OBJ=$BUILD/obj

mkdir -p "$OBJ" results dataset

if [ ! -f "$ROOT/src/rnnoise_data.c" ]; then
  echo "model weights missing; running download_model.sh"
  (cd "$ROOT" && ./download_model.sh)
fi

SRC="denoise rnn pitch kiss_fft celt_lpc nnet nnet_default \
parse_lpcnet_weights rnnoise_tables rnnoise_data"
for name in $SRC; do
  gcc -O2 -fPIC -I"$ROOT/include" -I"$ROOT/src" \
      -c "$ROOT/src/$name.c" -o "$OBJ/$name.o"
done
ar rcs "$BUILD/librnnoise.a" "$OBJ"/*.o

CFLAGS="-std=gnu11 -O2 -Wall -Wextra -I$ROOT/include -I$ROOT/src -I."
gcc $CFLAGS qa_common.c gen_dataset.c "$BUILD/librnnoise.a" -lm \
    -o "$BUILD/gen_dataset"
gcc $CFLAGS qa_common.c run_qa.c "$BUILD/librnnoise.a" -lm \
    -o "$BUILD/run_qa"
gcc $CFLAGS qa_common.c run_long.c "$BUILD/librnnoise.a" -lm \
    -o "$BUILD/run_long"

echo "== generating 500 deterministic noisy-speech samples =="
"$BUILD/gen_dataset" dataset results/manifest.csv

echo "== running 500-sample QA (clicks / latency / block size / determinism) =="
"$BUILD/run_qa" dataset results/per_sample.csv

echo "== running 3-hour continuous stability / drift test =="
"$BUILD/run_long" results/long_run.csv 1080000

echo
echo "===== QA SUMMARY ====="
cat results/summary.txt
echo
echo "===== LONG-RUN SUMMARY ====="
cat results/long_summary.txt
