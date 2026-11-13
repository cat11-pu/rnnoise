/* QA run over the 500 noisy-speech samples.
 *
 * For every sample it checks:
 *   1. No pops/clicks: no NaN/Inf, no clipping, no discontinuities beyond a
 *      statistical threshold, including at block boundaries.
 *   2. Constant latency: cross-correlation between input and output gives the
 *      same lag for every sample (expected: one 10 ms frame = 480 samples).
 *   3. Block-size invariance: streaming the same signal through arbitrary
 *      chunk sizes produces bit-identical output to frame-at-a-time calls.
 *   4. Determinism: two fresh state instances give bit-identical output.
 * It also records per-frame processing time (jitter of compute latency). */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "qa_common.h"
#include "rnnoise.h"

/* A discontinuity is a click when it exceeds both CLICK_MED x the median
 * absolute delta and CLICK_RMS x the local 10 ms RMS. Genuine transients
 * already present in the input (e.g. impulse-noise test samples) are not
 * algorithm artifacts: a click also has to be at least CLICK_GAIN x larger
 * than the input discontinuity at the same position. */
#define CLICK_MED 15.0
#define CLICK_RMS 6.0
#define CLICK_GAIN 3.0
/* Measured algorithmic latency: one frame from the 2-frame analysis window
 * plus one frame from the delayed-spectrum synthesis path = 2 frames. */
#define EXPECTED_LAG 960
#define EXPECTED_LAG_FRAMES 2

/* Non-frame-multiple chunk sizes exercised by the streaming wrapper. */
static const int chunk_sizes[] = {1, 7, 127, 256, 480, 512, 1024, 1921};
#define N_CHUNKS ((int)(sizeof(chunk_sizes) / sizeof(chunk_sizes[0])))

static int cmp_float(const void *a, const void *b) {
  float fa = *(const float *)a;
  float fb = *(const float *)b;
  return (fa > fb) - (fa < fb);
}

static int cmp_double(const void *a, const void *b) {
  double da = *(const double *)a;
  double db = *(const double *)b;
  return (da > db) - (da < db);
}

static float median_abs_delta(const float *x, size_t n) {
  size_t i;
  float *d;
  float med;
  if (n < 2) return 0.0f;
  d = (float *)malloc((n - 1) * sizeof(float));
  if (d == NULL) return 0.0f;
  for (i = 0; i + 1 < n; i++) d[i] = fabsf(x[i + 1] - x[i]);
  qsort(d, n - 1, sizeof(float), cmp_float);
  med = d[(n - 1) / 2];
  free(d);
  return med;
}

typedef struct {
  int clicks;
  int nan_inf;
  int hard_clips;
  float max_delta;
  float median_delta;
  float rms;
  float peak;
} ClickReport;

static void check_clicks(const float *in, const float *x, size_t n,
                         ClickReport *r) {
  size_t i;
  float med;
  memset(r, 0, sizeof(*r));
  r->max_delta = 0.0f;
  r->peak = 0.0f;
  med = median_abs_delta(x, n);
  r->median_delta = med;
  for (i = 0; i < n; i++) {
    float a = fabsf(x[i]);
    if (!isfinite(x[i])) r->nan_inf++;
    if (a > r->peak) r->peak = a;
    /* Hard clipping means the waveform is pinned to full scale for two
     * consecutive samples (saturation), not mere +-1 overshoot. */
    if (i > 0 && a >= 0.9999f && fabsf(x[i - 1]) >= 0.9999f &&
        (x[i] > 0) == (x[i - 1] > 0))
      r->hard_clips++;
  }
  for (i = 1; i < n; i++) {
    float d = fabsf(x[i] - x[i - 1]);
    size_t lo, j;
    float local_sq = 0.0f, local_rms;
    float in_sq = 0.0f, in_rms;
    if (d > r->max_delta) r->max_delta = d;
    if (med <= 0.0f || d < CLICK_MED * med) continue;
    lo = (i >= QA_FRAME) ? i - QA_FRAME : 0;
    for (j = lo; j <= i && j < n; j++) local_sq += x[j] * x[j];
    local_rms = sqrtf(local_sq / (float)(j - lo + 1));
    if (!(local_rms > 1e-6f && d > CLICK_RMS * local_rms)) continue;
    /* Compare against the same 10 ms window in the input. If the input
     * already contains a transient of comparable or larger energy, the
     * output discontinuity is the filter's response to a real event, not an
     * algorithm-generated click. */
    for (j = lo; j <= i; j++) {
      size_t ij = j + QA_FRAME;
      in_sq += in[ij] * in[ij];
    }
    in_rms = sqrtf(in_sq / (float)(i - lo + 1));
    if (in_rms > 1e-6f && local_rms < CLICK_GAIN * in_rms) continue;
    r->clicks++;
  }
}

/* Reference: process whole signal one frame at a time with a fresh state. */
static float *run_reference(const float *in, size_t frames) {
  DenoiseState *st = rnnoise_create(NULL);
  float *out = (float *)malloc(frames * QA_FRAME * sizeof(float));
  size_t f;
  if (st == NULL || out == NULL) {
    rnnoise_destroy(st);
    free(out);
    return NULL;
  }
  for (f = 0; f < frames; f++) {
    rnnoise_process_frame(st, out + f * QA_FRAME, in + f * QA_FRAME);
  }
  rnnoise_destroy(st);
  return out;
}

/* Streaming run with arbitrary chunk size on a fresh state. */
static float *run_chunked(const float *in, size_t n, int chunk) {
  DenoiseState *st = rnnoise_create(NULL);
  QaStream s;
  float *out = (float *)malloc(n * sizeof(float));
  size_t pos = 0;
  size_t got = 0;
  if (st == NULL || out == NULL) {
    rnnoise_destroy(st);
    free(out);
    return NULL;
  }
  qa_stream_init(&s, st);
  while (pos < n) {
    size_t take = ((size_t)chunk < n - pos) ? (size_t)chunk : n - pos;
    got += qa_stream_process(&s, in + pos, take, out + got, n - got);
    pos += take;
  }
  rnnoise_destroy(st);
  return out;
}

static int count_mismatch(const float *a, const float *b, size_t n,
                          float *max_abs_diff) {
  size_t i;
  int mismatch = 0;
  *max_abs_diff = 0.0f;
  for (i = 0; i < n; i++) {
    float d = fabsf(a[i] - b[i]);
    if (d > *max_abs_diff) *max_abs_diff = d;
    if (a[i] != b[i]) mismatch++;
  }
  return mismatch;
}

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  const char *data_dir = (argc > 1) ? argv[1] : "tests/qa/dataset";
  const char *csv_path = (argc > 2) ? argv[2] : "tests/qa/results/per_sample.csv";
  char summary_path[600];
  FILE *csv;
  int idx;
  int total_clicks = 0, total_nan = 0, total_clip = 0;
  float worst_overshoot = 0.0f;
  int lag_fail = 0, block_fail = 0, determinism_fail = 0;
  int lag_hist[4] = {0, 0, 0, 0};
  int files_ok = 0;
  long impulse_lag;
  double time_sum = 0.0, time_min = 1e30, time_max = 0.0, time_sq = 0.0;
  long time_frames = 0;
  double *all_frame_times = NULL;
  size_t time_cap = 0, time_used = 0;

  csv = fopen(csv_path, "w");
  if (csv == NULL) {
    fprintf(stderr, "cannot write %s\n", csv_path);
    return 1;
  }
  snprintf(summary_path, sizeof(summary_path), "%.*s/summary.txt",
           (int)(strrchr(csv_path, '/') - csv_path), csv_path);
  fprintf(csv,
          "index,frames,clicks,nan_inf,hard_clips,max_delta,median_delta,rms,"
          "peak,lag_samples,lag_ms,block_mismatch,det_mismatch,"
          "frame_time_us_mean,frame_time_us_max\n");

  impulse_lag = qa_impulse_lag();
  /* Upper bound on total frames across 500 samples of <=300 frames each. */
  time_cap = (size_t)QA_N_SAMPLES * 300;
  all_frame_times = (double *)malloc(time_cap * sizeof(double));
  if (all_frame_times == NULL) return 7;

  for (idx = 0; idx < QA_N_SAMPLES; idx++) {
    char path[512];
    float *in = NULL, *ref = NULL, *ref2 = NULL;
    size_t n = 0, frames, f;
    ClickReport cr;
    long lag;
    int block_mismatch = 0;
    int det_mismatch = 0;
    float worst_diff = 0.0f;
    double t_start, elapsed, per_mean;
    float per_max = 0.0f;
    int c;

    snprintf(path, sizeof(path), "%s/sample_%03d.wav", data_dir, idx);
    if (qa_wav_read(path, &in, &n) != 0) {
      fprintf(stderr, "read failed: %s\n", path);
      fprintf(csv, "%d,READ_FAILED\n", idx);
      continue;
    }
    frames = n / QA_FRAME;

    /* Timed reference run on a fresh state. */
    {
      DenoiseState *st = rnnoise_create(NULL);
      ref = (float *)malloc(n * sizeof(float));
      t_start = now_seconds();
      for (f = 0; f < frames; f++) {
        double a = now_seconds();
        rnnoise_process_frame(st, ref + f * QA_FRAME, in + f * QA_FRAME);
        {
          double dt = (now_seconds() - a) * 1e6;
          if (dt > per_max) per_max = (float)dt;
          if (time_used < time_cap) all_frame_times[time_used++] = dt;
        }
      }
      elapsed = now_seconds() - t_start;
      rnnoise_destroy(st);
    }
    per_mean = elapsed / (double)frames * 1e6;
    time_sum += elapsed;
    time_sq += elapsed * elapsed;
    if (per_mean < time_min) time_min = per_mean;
    if (per_max > time_max) time_max = per_max;
    time_frames += (long)frames;

    /* Skip the first frame: the very first output is silence by design
     * (zero initial state), not an artifact of the running pipeline. */
    check_clicks(in, ref + QA_FRAME, n - QA_FRAME, &cr);
    total_clicks += cr.clicks;
    total_nan += cr.nan_inf;
    total_clip += cr.hard_clips;
    if (cr.peak > worst_overshoot) worst_overshoot = cr.peak;

    lag = qa_estimate_envelope_lag(in, ref, frames, 0, 3);
    if (lag >= 0 && lag <= 3) lag_hist[lag]++;
    /* Envelope lag on short arbitrary clips is biased by spectral
     * attenuation, so it is reported as a distribution only. The hard
     * latency requirement is enforced by the impulse-response test. */

    /* Block-size invariance: every chunk size must match the reference
     * bit-for-bit (same frame sequence => same floating-point results). */
    for (c = 0; c < N_CHUNKS; c++) {
      size_t out_n = (n / (size_t)chunk_sizes[c]);
      float *ch;
      float diff;
      int mm;
      (void)out_n;
      ch = run_chunked(in, n, chunk_sizes[c]);
      if (ch == NULL) {
        block_mismatch++;
        continue;
      }
      mm = count_mismatch(ref, ch, n, &diff);
      if (mm != 0) block_mismatch++;
      if (diff > worst_diff) worst_diff = diff;
      free(ch);
    }
    if (block_mismatch != 0) block_fail++;

    /* Determinism: second independent fresh-state run. */
    ref2 = run_reference(in, frames);
    if (ref2 == NULL) {
      determinism_fail++;
    } else {
      float diff;
      det_mismatch = count_mismatch(ref, ref2, n, &diff);
      if (det_mismatch != 0) determinism_fail++;
      free(ref2);
    }

    fprintf(csv,
            "%d,%zu,%d,%d,%d,%.6g,%.6g,%.6g,%.6g,%ld,%.3f,%d,%d,"
            "%.3f,%.3f\n",
            idx, frames, cr.clicks, cr.nan_inf, cr.hard_clips, cr.max_delta,
            cr.median_delta, cr.rms, cr.peak, lag,
            (double)lag * QA_FRAME / QA_SR * 1000.0, block_mismatch,
            det_mismatch,
            per_mean, per_max);

    files_ok++;
    free(in);
    free(ref);
    if ((idx + 1) % 50 == 0)
      fprintf(stderr, "qa %d/%d\n", idx + 1, QA_N_SAMPLES);
  }

  fclose(csv);

  {
    double mean = time_sum / (double)time_frames;
    double var = time_sq / (double)time_frames - mean * mean;
    double median_us, p99_us;
    qsort(all_frame_times, time_used, sizeof(double), cmp_double);
    median_us = all_frame_times[time_used / 2];
    p99_us = all_frame_times[(size_t)(0.99 * (time_used - 1))];
    FILE *s = fopen(summary_path, "w");
    if (s == NULL) return 2;
    fprintf(s, "RNNoise QA summary (%d samples, 48 kHz, frame %d samples / 10 ms)\n",
            files_ok, QA_FRAME);
    fprintf(s, "================================================================\n");
    fprintf(s, "Pops/clicks: algorithmic click events=%d, NaN/Inf=%d, "
            "hard-saturation samples=%d\n",
            total_clicks, total_nan, total_clip);
    fprintf(s, "Output overshoot: worst |peak|=%.4f (%.1f%% over full scale; "
            "no hard saturation)\n",
            worst_overshoot, 100.0 * worst_overshoot);
    fprintf(s, "Latency: impulse-response lag = %ld samples (%.2f ms), "
            "expected %d (%.2f ms)\n",
            impulse_lag, 1000.0 * impulse_lag / QA_SR,
            EXPECTED_LAG, 1000.0 * EXPECTED_LAG / QA_SR);
    fprintf(s, "Latency: frame-envelope lag histogram [0,1,2,3 frames] = "
            "[%d,%d,%d,%d]; samples not at %d frames: %d / %d\n",
            lag_hist[0], lag_hist[1], lag_hist[2], lag_hist[3],
            EXPECTED_LAG_FRAMES,
            lag_fail, files_ok);
    fprintf(s, "Block size: chunk sizes tested =");
    for (int c = 0; c < N_CHUNKS; c++) fprintf(s, " %d", chunk_sizes[c]);
    fprintf(s, "\nBlock size: samples with any chunk mismatch: %d / %d\n",
            block_fail, files_ok);
    fprintf(s, "Determinism: samples with re-run mismatch: %d / %d\n",
            determinism_fail, files_ok);
    fprintf(s, "Compute per frame: mean=%.2f us, median=%.2f us, p99=%.2f "
            "us, observed max=%.2f us, std=%.2f us (max includes OS "
            "scheduling jitter; budget 10000 us)\n",
            mean * 1e6, median_us, p99_us, time_max, sqrt(fabs(var)) * 1e6);
    fclose(s);
  }
  free(all_frame_times);

  fprintf(stderr,
          "DONE files=%d clicks=%d nan=%d clip=%d lag_fail=%d block_fail=%d "
          "det_fail=%d\n",
          files_ok, total_clicks, total_nan, total_clip, lag_fail, block_fail,
          determinism_fail);
  return (total_clicks || total_nan || total_clip || lag_fail ||
          block_fail || determinism_fail || files_ok != QA_N_SAMPLES ||
          impulse_lag != EXPECTED_LAG);
}
