/* Stand-alone PCM sample-rate converter using the same streaming resampler
   as the denoise pipeline.
   Usage: resample_pcm <in s16le> <in_rate> <out s16le> <out_rate> */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "resample.h"

#define HALF_WIDTH 24
#define KAISER_BETA 8.6

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr,
            "usage: %s <in s16le> <in_rate> <out s16le> <out_rate>\n",
            argv[0]);
    return 1;
  }
  int in_rate = atoi(argv[2]);
  int out_rate = atoi(argv[4]);
  FILE *fin = fopen(argv[1], "rb");
  FILE *fout = fopen(argv[3], "wb");
  if (!fin || !fout) { perror("fopen"); return 1; }

  fseek(fin, 0, SEEK_END);
  long bytes = ftell(fin);
  fseek(fin, 0, SEEK_SET);
  size_t n = (size_t) bytes / sizeof(short);
  short *raw = malloc(bytes);
  if (fread(raw, sizeof(short), n, fin) != n) {
    fprintf(stderr, "short read\n");
    return 1;
  }
  float *in = malloc(n * sizeof(float));
  for (size_t i = 0; i < n; i++) in[i] = (float) raw[i];

  Resampler *r = resampler_create(in_rate, out_rate,
                                  HALF_WIDTH, KAISER_BETA);
  size_t cap = resampler_max_out(r, n) + 2 * HALF_WIDTH + 4;
  float *out = malloc(cap * sizeof(float));
  size_t on = 0;
  resampler_process(r, in, n, out, &on, cap);
  resampler_flush(r, out, &on, cap);

  /* Compensate the filter group delay and match the ideal output length. */
  double delay = resampler_delay_out(r);
  size_t skip = (size_t) (delay + 0.5);
  size_t ideal = (size_t) ((double) n * out_rate / in_rate + 0.5);
  if (skip + ideal > on) skip = (on > ideal) ? on - ideal : 0;
  if (ideal > on - skip) ideal = on - skip;
  short *out_raw = malloc(ideal * sizeof(short));
  for (size_t i = 0; i < ideal; i++) {
    float v = out[skip + i];
    if (v > 32767.0f) v = 32767.0f;
    if (v < -32768.0f) v = -32768.0f;
    out_raw[i] = (short) lrintf(v);
  }
  fwrite(out_raw, sizeof(short), ideal, fout);

  free(out_raw);
  free(raw);
  free(in);
  free(out);
  resampler_destroy(r);
  fclose(fin);
  fclose(fout);
  return 0;
}
