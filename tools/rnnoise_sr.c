/* Sample-rate aware RNNoise front end.

   Usage: rnnoise_sr <in.pcm_s16le> <out.pcm_s16le> <rate_in> [rate_out]

   Accepts 16000, 44100 and 48000 Hz mono 16-bit PCM. Non-48k inputs are
   resampled to 48 kHz for the RNNoise model and resampled back to the
   original rate, so duration and pitch are preserved. Any other rate is
   rejected instead of being silently treated as 48 kHz (which used to
   transpose the signal). */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rnnoise.h"
#include "resample.h"

#define MODEL_RATE 48000
#define FRAME_SIZE 480
#define HALF_WIDTH 24
#define KAISER_BETA 8.6

static void write_clamped(FILE *fout, const float *x, size_t n,
                          short *tmp) {
  for (size_t i = 0; i < n; i++) {
    float v = x[i];
    if (v > 32767.0f) v = 32767.0f;
    if (v < -32768.0f) v = -32768.0f;
    tmp[i] = (short) lrintf(v);
  }
  fwrite(tmp, sizeof(short), n, fout);
}

static int supported_rate(int rate) {
  return rate == 16000 || rate == 44100 || rate == 48000;
}

int main(int argc, char **argv) {
  if (argc < 4 || argc > 5) {
    fprintf(stderr,
            "usage: %s <in s16le> <out s16le> <rate_in> [rate_out]\n",
            argv[0]);
    return 1;
  }
  const char *in_path = argv[1];
  const char *out_path = argv[2];
  int rate_in = atoi(argv[3]);
  int rate_out = (argc == 5) ? atoi(argv[4]) : rate_in;
  if (!supported_rate(rate_in) || !supported_rate(rate_out)) {
    fprintf(stderr,
            "error: only 16000, 44100 and 48000 Hz are supported "
            "(got in=%d out=%d)\n", rate_in, rate_out);
    return 2;
  }

  FILE *fin = fopen(in_path, "rb");
  if (!fin) { perror("fopen input"); return 1; }
  FILE *fout = fopen(out_path, "wb");
  if (!fout) { perror("fopen output"); fclose(fin); return 1; }

  DenoiseState *st = rnnoise_create(NULL);
  if (!st) {
    fprintf(stderr, "error: rnnoise_create failed\n");
    return 1;
  }

  int rc = 0;
  size_t input_samples = 0;
  if (rate_in == MODEL_RATE && rate_out == MODEL_RATE) {
    /* Native path: frame-exact, no resampling. */
    short tmp[FRAME_SIZE];
    float frame[FRAME_SIZE];
    for (;;) {
      size_t got = fread(tmp, sizeof(short), FRAME_SIZE, fin);
      if (got == 0) break;
      input_samples += got;
      if (got < FRAME_SIZE) {
        for (size_t i = got; i < FRAME_SIZE; i++) tmp[i] = 0;
      }
      for (int i = 0; i < FRAME_SIZE; i++) frame[i] = (float) tmp[i];
      rnnoise_process_frame(st, frame, frame);
      write_clamped(fout, frame, got, tmp);
    }
  } else {
    Resampler *up = resampler_create(rate_in, MODEL_RATE,
                                     HALF_WIDTH, KAISER_BETA);
    Resampler *down = resampler_create(MODEL_RATE, rate_out,
                                       HALF_WIDTH, KAISER_BETA);
    if (!up || !down) {
      fprintf(stderr, "error: resampler_create failed\n");
      rc = 1;
      goto cleanup;
    }

    /* Whole-file streaming: collect the final-rate output, then trim to the
       exact input duration (the filter group delay otherwise pads the
       signal by ~half_width samples on each side). */
    fseek(fin, 0, SEEK_END);
    long in_bytes = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    input_samples = (size_t) in_bytes / sizeof(short);
    size_t in_chunk = (size_t) (rate_in * 20 / 1000);
    if (in_chunk == 0) in_chunk = 1;

    short *in_raw = malloc(in_chunk * sizeof(short));
    float *in_buf = malloc(in_chunk * sizeof(float));
    float *frame = malloc(FRAME_SIZE * sizeof(float));

    /* Upper bounds: up-resampled 48k and final-rate output counts. */
    size_t up_total_cap = resampler_max_out(up, input_samples) +
                          2 * HALF_WIDTH + FRAME_SIZE + 4;
    size_t out_total_cap = resampler_max_out(down, up_total_cap) +
                           2 * HALF_WIDTH + FRAME_SIZE + 4;
    float *up_all = malloc(up_total_cap * sizeof(float));
    float *out_all = malloc(out_total_cap * sizeof(float));
    short *out_tmp = malloc((input_samples + FRAME_SIZE) * sizeof(short));
    float *down_tmp = malloc((resampler_max_out(down, FRAME_SIZE) +
                              FRAME_SIZE) * sizeof(float));
    if (!in_raw || !in_buf || !frame || !up_all || !out_all ||
        !out_tmp || !down_tmp) {
      fprintf(stderr, "error: out of memory\n");
      rc = 1;
      free(in_raw); free(in_buf); free(frame); free(up_all);
      free(out_all); free(out_tmp); free(down_tmp);
      goto cleanup_res;
    }

    size_t up_total = 0;
    for (;;) {
      size_t got = fread(in_raw, sizeof(short), in_chunk, fin);
      if (got == 0) break;
      for (size_t i = 0; i < got; i++) in_buf[i] = (float) in_raw[i];
      resampler_process(up, in_buf, got, up_all, &up_total, up_total_cap);
      if (got < in_chunk) break;
    }
    resampler_flush(up, up_all, &up_total, up_total_cap);

    /* Denoise exact 480-sample frames and down-resample on the fly. */
    size_t out_total = 0;
    size_t full_frames = up_total / FRAME_SIZE;
    for (size_t f = 0; f < full_frames; f++) {
      memcpy(frame, up_all + f * FRAME_SIZE,
             FRAME_SIZE * sizeof(float));
      rnnoise_process_frame(st, frame, frame);
      size_t down_n = 0;
      resampler_process(down, frame, FRAME_SIZE,
                        down_tmp, &down_n,
                        resampler_max_out(down, FRAME_SIZE) + FRAME_SIZE);
      for (size_t i = 0; i < down_n; i++) {
        if (out_total >= out_total_cap) break;
        out_all[out_total++] = down_tmp[i];
      }
    }
    /* Last partial 48k frame, zero-padded. */
    size_t rem = up_total - full_frames * FRAME_SIZE;
    if (rem > 0) {
      memcpy(frame, up_all + full_frames * FRAME_SIZE,
             rem * sizeof(float));
      memset(frame + rem, 0, (FRAME_SIZE - rem) * sizeof(float));
      rnnoise_process_frame(st, frame, frame);
      size_t down_n = 0;
      resampler_process(down, frame, FRAME_SIZE,
                        down_tmp, &down_n,
                        resampler_max_out(down, FRAME_SIZE) + FRAME_SIZE);
      for (size_t i = 0; i < down_n && out_total < out_total_cap; i++) {
        out_all[out_total++] = down_tmp[i];
      }
    }
    resampler_flush(down, out_all, &out_total, out_total_cap);

    /* The two cascaded converters delay the signal; align by skipping the
       rounded combined group delay and keep exactly input_samples samples. */
    double delay = resampler_delay_out(up) *
                   ((double) rate_out / MODEL_RATE) +
                   resampler_delay_out(down);
    size_t skip = (size_t) (delay + 0.5);
    if (skip + input_samples > out_total) {
      skip = (out_total > input_samples) ? out_total - input_samples : 0;
    }
    write_clamped(fout, out_all + skip, input_samples, out_tmp);

    free(in_raw); free(in_buf); free(frame); free(up_all);
    free(out_all); free(out_tmp); free(down_tmp);
cleanup_res:
    resampler_destroy(up);
    resampler_destroy(down);
  }

cleanup:
  rnnoise_destroy(st);
  if (fclose(fout) != 0) perror("fclose output");
  fclose(fin);
  return rc;
}
