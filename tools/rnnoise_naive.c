/* Reproduces the original, rate-unaware behavior: every input is treated as
   48 kHz and pushed through RNNoise in 480-sample frames. Feeding 16 kHz or
   44.1 kHz audio this way transposes pitch/speed (the reported bug).
   Kept as the "before" baseline for the comparison. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "rnnoise.h"

#define FRAME_SIZE 480

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <in s16le> <out s16le>\n", argv[0]);
    return 1;
  }
  FILE *fin = fopen(argv[1], "rb");
  FILE *fout = fopen(argv[2], "wb");
  if (!fin || !fout) { perror("fopen"); return 1; }
  DenoiseState *st = rnnoise_create(NULL);

  int first = 1;
  short tmp[FRAME_SIZE];
  float x[FRAME_SIZE];
  for (;;) {
    size_t got = fread(tmp, sizeof(short), FRAME_SIZE, fin);
    if (got == 0) break;
    if (got < FRAME_SIZE) {
      for (size_t i = got; i < FRAME_SIZE; i++) tmp[i] = 0;
    }
    for (int i = 0; i < FRAME_SIZE; i++) x[i] = (float) tmp[i];
    rnnoise_process_frame(st, x, x);
    for (int i = 0; i < FRAME_SIZE; i++) {
      float v = x[i];
      if (v > 32767.0f) v = 32767.0f;
      if (v < -32768.0f) v = -32768.0f;
      tmp[i] = (short) lrintf(v);
    }
    if (!first) fwrite(tmp, sizeof(short), FRAME_SIZE, fout);
    first = 0;
  }
  rnnoise_destroy(st);
  fclose(fin);
  fclose(fout);
  return 0;
}
