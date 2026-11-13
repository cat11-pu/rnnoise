/* Long-duration stability / drift test.
 *
 * One DenoiseState processes a continuous stream for LONG_FRAMES frames
 * (default 1 080 000 frames = 3 h of audio). At regular checkpoints the
 * output of the long-running state is compared against a freshly created
 * state fed the same window of input. A stateful filter on a stationary
 * looped source must converge to the same response at every checkpoint, so
 * any growing difference is evidence of drift. The run also tracks NaN/Inf,
 * clipping, output RMS stability and per-frame compute time. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "qa_common.h"
#include "rnnoise.h"

#define DEFAULT_LONG_FRAMES 1080000L      /* 3 h */
#define LOOP_FRAMES 3000                  /* 30 s looped source */
#define CHECK_EVERY_FRAMES 36000L         /* every 10 min */
#define WARMUP_FRAMES 18000L              /* 3 min convergence */
#define WINDOW_FRAMES 600                 /* 6 s comparison window */
#define FRESH_WARMUP 18000L               /* history replayed into fresh state */

static long long mod_pos(long long a, long long m) {
  long long r = a % m;
  return r < 0 ? r + m : r;
}

static int cmp_double(const void *a, const void *b) {
  double da = *(const double *)a;
  double db = *(const double *)b;
  return (da > db) - (da < db);
}

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Stationary loop: continuous pink-ish noise + low-level tone + slow
 * amplitude swell, generated once and repeated. */
static void make_loop(float *buf, size_t n) {
  QaRng rng;
  QaWhiteSrc white;
  double pink[7] = {0};
  size_t i;
  qa_rng_seed(&rng, 0xCAFEF00DULL);
  qa_white_init(&white, 0x12345678ULL);
  for (i = 0; i < n; i++) {
    double t = (double)i / QA_SR;
    double w = qa_white_next(&white);
    double v;
    pink[0] = 0.99886 * pink[0] + w * 0.0555179;
    pink[1] = 0.99332 * pink[1] + w * 0.0750759;
    pink[2] = 0.96900 * pink[2] + w * 0.1538520;
    pink[3] = 0.86650 * pink[3] + w * 0.3104856;
    pink[4] = 0.55000 * pink[4] + w * 0.5329522;
    pink[5] = -0.7616 * pink[5] - w * 0.0168980;
    v = (pink[0] + pink[1] + pink[2] + pink[3] + pink[4] + pink[5] +
         pink[6] + w * 0.5362) * 0.11;
    pink[6] = w * 0.115926;
    v += 0.05 * sin(2.0 * QA_PI * 440.0 * t);
    v *= 0.85 + 0.15 * sin(2.0 * QA_PI * 0.1 * t);
    buf[i] = (float)(v * 0.5);
  }
}

int main(int argc, char **argv) {
  long long total_frames = DEFAULT_LONG_FRAMES;
  const char *csv_path = (argc > 1) ? argv[1]
                                    : "tests/qa/results/long_run.csv";
  char summary_path[600];
  FILE *csv;
  DenoiseState *st;
  float *loop;
  float *ring;                 /* recent outputs, WINDOW_FRAMES deep */
  float frame_in[QA_FRAME], frame_out[QA_FRAME];
  long long f;
  long long nan_inf = 0, clips = 0;
  double time_sum = 0.0, time_max = 0.0;
  double *time_samples;
  long long time_cap = 0;
  float global_peak = 0.0f;
  float drift_max = 0.0f, rms_diff_max = 0.0f;
  long checkpoints = 0;

  if (argc > 2) total_frames = atoll(argv[2]);
  snprintf(summary_path, sizeof(summary_path), "%.*s/long_summary.txt",
           (int)(strrchr(csv_path, '/') - csv_path), csv_path);

  loop = (float *)malloc((size_t)LOOP_FRAMES * QA_FRAME * sizeof(float));
  ring = (float *)malloc((size_t)WINDOW_FRAMES * QA_FRAME * sizeof(float));
  time_samples = (double *)malloc((size_t)total_frames * sizeof(double));
  if (loop == NULL || ring == NULL || time_samples == NULL) return 2;
  memset(ring, 0, (size_t)WINDOW_FRAMES * QA_FRAME * sizeof(float));
  make_loop(loop, (size_t)LOOP_FRAMES * QA_FRAME);

  st = rnnoise_create(NULL);
  if (st == NULL) return 3;

  csv = fopen(csv_path, "w");
  if (csv == NULL) return 4;
  fprintf(csv,
          "elapsed_s,frames,fresh_maxdiff,fresh_rms_diff,window_rms,"
          "window_peak,nan_inf_total,clips_total,frame_time_us_mean\n");

  for (f = 0; f < total_frames; f++) {
    const float *src;
    double a;
    int i;
    long long loop_pos;
    double dt;

    loop_pos = (f % LOOP_FRAMES) * QA_FRAME;
    src = loop + loop_pos;
    memcpy(frame_in, src, QA_FRAME * sizeof(float));

    a = now_seconds();
    rnnoise_process_frame(st, frame_out, frame_in);
    dt = (now_seconds() - a) * 1e6;
    time_sum += dt;
    if (dt > time_max) time_max = dt;
    if (time_cap < total_frames) time_samples[time_cap++] = dt;

    memcpy(ring + (f % WINDOW_FRAMES) * QA_FRAME, frame_out,
           QA_FRAME * sizeof(float));
    for (i = 0; i < QA_FRAME; i++) {
      float v = frame_out[i];
      if (!isfinite(v)) nan_inf++;
      if (fabsf(v) >= 0.999f) clips++;
      if (fabsf(v) > global_peak) global_peak = fabsf(v);
    }

    if (f >= WARMUP_FRAMES &&
        ((f - WARMUP_FRAMES) % CHECK_EVERY_FRAMES) == 0) {
      DenoiseState *fresh = rnnoise_create(NULL);
      float max_diff = 0.0f;
      double long_sq = 0.0, fresh_sq = 0.0;
      float win_peak = 0.0f;
      int w, j;
      if (fresh == NULL) return 5;
      /* Replay FRESH_WARMUP history frames through the fresh state so it
       * reaches the same steady state, then compare the last
       * WINDOW_FRAMES frames. The deterministic loop makes exact replay
       * possible. */
      for (w = 0; w < FRESH_WARMUP + WINDOW_FRAMES; w++) {
        long long idx = f - WINDOW_FRAMES + 1 + w - FRESH_WARMUP;
        float in2[QA_FRAME], out2[QA_FRAME];
        const float *src2 = loop + mod_pos(idx, LOOP_FRAMES) * QA_FRAME;
        memcpy(in2, src2, QA_FRAME * sizeof(float));
        rnnoise_process_frame(fresh, out2, in2);
        if (w < FRESH_WARMUP) continue;
        {
          long long ref_idx = idx;
          const float *ref_out =
              ring + mod_pos(ref_idx, WINDOW_FRAMES) * QA_FRAME;
          for (j = 0; j < QA_FRAME; j++) {
            float d = fabsf(out2[j] - ref_out[j]);
            if (d > max_diff) max_diff = d;
            fresh_sq += (double)out2[j] * out2[j];
            long_sq += (double)ref_out[j] * ref_out[j];
            if (fabsf(ref_out[j]) > win_peak) win_peak = fabsf(ref_out[j]);
          }
        }
      }
      rnnoise_destroy(fresh);
      {
        size_t cnt = (size_t)WINDOW_FRAMES * QA_FRAME;
        float long_rms = (float)sqrt(long_sq / (double)cnt);
        float fresh_rms = (float)sqrt(fresh_sq / (double)cnt);
        float rms_diff = fabsf(long_rms - fresh_rms);
        if (max_diff > drift_max) drift_max = max_diff;
        if (rms_diff > rms_diff_max) rms_diff_max = rms_diff;
        checkpoints++;
        fprintf(csv,
                "%.1f,%lld,%.8g,%.8g,%.8g,%.6g,%lld,%lld,%.3f\n",
                (double)f * QA_FRAME / QA_SR, f, max_diff, rms_diff,
                long_rms, win_peak, nan_inf, clips,
                time_sum / (double)(f + 1));
      }
    }
  }

  fclose(csv);
  rnnoise_destroy(st);

  {
    double median_us, p99_us;
    qsort(time_samples, (size_t)time_cap, sizeof(double),
          cmp_double);
    median_us = time_samples[time_cap / 2];
    p99_us = time_samples[(long long)(0.99 * (time_cap - 1))];
    FILE *s = fopen(summary_path, "w");
    if (s == NULL) {
      free(loop);
      free(ring);
      return 6;
    }
    fprintf(s, "RNNoise long-run stability\n");
    fprintf(s, "==========================\n");
    fprintf(s, "duration: %.2f h (%lld frames, %lld samples)\n",
            (double)total_frames * QA_FRAME / QA_SR / 3600.0, total_frames,
            total_frames * QA_FRAME);
    fprintf(s, "NaN/Inf samples: %lld\n", nan_inf);
    fprintf(s, "clipping samples (|x|>=0.999): %lld\n", clips);
    fprintf(s, "global output peak: %.6f\n", global_peak);
    fprintf(s, "checkpoints: %ld (every %.0f min after %.0f min warmup)\n",
            checkpoints,
            (double)CHECK_EVERY_FRAMES * QA_FRAME / QA_SR / 60.0,
            (double)WARMUP_FRAMES * QA_FRAME / QA_SR / 60.0);
    fprintf(s, "drift vs fresh converged state: max |diff|=%.3g, "
            "max RMS diff=%.3g\n", drift_max, rms_diff_max);
    fprintf(s, "per-frame time: mean=%.2f us, median=%.2f us, "
            "p99=%.2f us, observed max=%.2f us (max includes OS scheduling "
            "jitter; budget 10000 us)\n",
            time_sum / (double)total_frames, median_us, p99_us, time_max);
    fclose(s);
  }

  fprintf(stderr,
          "LONG DONE frames=%lld nan=%lld clips=%lld peak=%.4f "
          "drift_max=%.3g mean_us=%.2f median_us=%.2f p99_us=%.2f\n",
          total_frames, nan_inf, clips, global_peak, drift_max,
          time_sum / (double)total_frames,
          time_samples[time_cap / 2],
          time_samples[(long long)(0.99 * (time_cap - 1))]);
  free(loop);
  free(ring);
  free(time_samples);
  return (nan_inf || clips) ? 1 : 0;
}
