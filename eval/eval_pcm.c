/* Objective quality / artifact evaluator for mono s16le PCM files.

   Metrics reported per file:
     - segmental SNR (dB) and a MOS-like 1..5 score mapped from it
     - pop/click count: inter-sample jumps exceeding an adaptive threshold
     - hard clipping count
     - duration in seconds and sample count

   Usage:
     eval_pcm <ref s16le> <ref_rate> <test s16le> <test_rate> [label]

   References and tests at different rates are aligned in time before
   comparison (linear interpolation), so 16/44.1/48k outputs can be judged
   against the same 48 kHz reference. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SEG_MS 32
#define EPS 1e-8f
#define LPF_HW 32

typedef struct {
  float *x;
  size_t n;
  int rate;
} Signal;

static int load_pcm(const char *path, int rate, Signal *s) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); return -1; }
  fseek(f, 0, SEEK_END);
  long bytes = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (bytes <= 0 || (bytes % (long) sizeof(short)) != 0) {
    fprintf(stderr, "error: %s is not 16-bit PCM\n", path);
    fclose(f);
    return -1;
  }
  size_t n = (size_t) bytes / sizeof(short);
  short *raw = malloc(bytes);
  if (!raw || fread(raw, sizeof(short), n, f) != n) {
    fprintf(stderr, "error: cannot read %s\n", path);
    free(raw);
    fclose(f);
    return -1;
  }
  fclose(f);
  s->x = malloc(n * sizeof(float));
  s->n = n;
  s->rate = rate;
  for (size_t i = 0; i < n; i++) s->x[i] = (float) raw[i];
  free(raw);
  return 0;
}

/* Sample signal s at fractional time t (seconds), zero outside the range. */
static float sample_at(const Signal *s, double t) {
  double pos = t * s->rate;
  if (pos < 0 || pos >= (double) s->n - 1.0) return 0.0f;
  size_t i = (size_t) pos;
  double frac = pos - (double) i;
  return (float) ((1.0 - frac) * s->x[i] + frac * s->x[i + 1]);
}

/* Zero-phase windowed-sinc low pass, used to compare only the frequency
   band that a narrow-band input can physically represent. */
static void low_pass(const Signal *s, double cutoff_hz, Signal *out) {
  out->n = s->n;
  out->rate = s->rate;
  out->x = calloc(s->n, sizeof(float));
  double norm = 2.0 * cutoff_hz / s->rate;
  for (size_t i = 0; i < s->n; i++) {
    double acc = 0;
    for (int k = -LPF_HW; k <= LPF_HW; k++) {
      long j = (long) i + k;
      if (j < 0 || (size_t) j >= s->n) continue;
      double arg = M_PI * norm * k;
      double sinc = (k == 0) ? 1.0 : sin(arg) / arg;
      acc += s->x[j] * norm * sinc;
    }
    out->x[i] = (float) acc;
  }
}

static float rms(const Signal *s) {
  double acc = 0;
  for (size_t i = 0; i < s->n; i++) acc += (double) s->x[i] * s->x[i];
  return (float) sqrt(acc / (double) s->n);
}

/* Segment-SNR -> MOS mapping. The logistic spans the useful range:
   ~10 dB -> 2.0, ~20 dB -> 3.0, ~30 dB -> 4.0, 40+ dB -> ~4.5. */
static float snr_to_mos(float seg_snr_db) {
  float mos = 1.0f + 4.0f / (1.0f + expf(-(seg_snr_db - 20.0f) / 4.0f));
  if (mos > 4.5f) mos = 4.5f;
  return mos;
}

/* Relative-quality score used for the "vs 48 kHz baseline" comparison.
   Both signals are denoiser outputs at the same nominal content, so a high
   SNR means the resampling path is transparent. We anchor a perfect match
   at 4.5 and deduct from the error energy relative to the reference energy,
   which is stable across rates and signal levels. */
static float relative_mos(double signal_e, double error_e) {
  double snr = 10.0 * log10((signal_e + 1e-9) / (error_e + 1e-9));
  if (snr >= 40.0) return 4.5f;
  if (snr <= 0.0) return 1.0f;
  return (float) (1.0 + 3.5 * snr / 40.0);
}

static void evaluate(const Signal *ref, const Signal *test,
                     float *seg_snr_out, float *mos_out,
                     long *pops_out, long *clips_out) {
  int step = test->rate * SEG_MS / 1000;
  if (step < 1) step = 1;
  double snr_sum = 0;
  long seg_count = 0;

  for (size_t start = 0; start + (size_t) step <= test->n;
       start += (size_t) step) {
    double noise_e = 0, sig_e = 0;
    for (int k = 0; k < step; k++) {
      size_t idx = start + (size_t) k;
      double t = (double) idx / test->rate;
      float r = sample_at(ref, t);
      float e = test->x[idx] - r;
      noise_e += (double) e * e;
      sig_e += (double) r * r;
    }
    if (sig_e < 1e4) continue;                 /* skip silence segments */
    double snr = 10.0 * log10((sig_e + EPS) / (noise_e + EPS));
    if (snr > 60.0) snr = 60.0;
    if (snr < -10.0) snr = -10.0;
    snr_sum += snr;
    seg_count++;
  }
  float seg_snr = seg_count ? (float) (snr_sum / seg_count) : 0.0f;
  *seg_snr_out = seg_snr;
  *mos_out = snr_to_mos(seg_snr);

  /* Pop/click detection. A real click is an isolated, one-sample impulse:
     both edges are large and point back toward the local trend, so the
     center sample stands out from BOTH neighbors. Ordinary voiced-speech
     zero crossings move in one direction and do not satisfy this. The
     threshold is max(25% full scale, 8x the mean absolute step). */
  long pops = 0, clips = 0;
  double abs_sum = 0;
  for (size_t i = 1; i < test->n; i++) {
    abs_sum += fabsf(test->x[i] - test->x[i - 1]);
  }
  float avg_step = (float) (abs_sum / (double) (test->n - 1));
  float thresh = 8.0f * avg_step;
  if (thresh < 8192.0f) thresh = 8192.0f;      /* >= 25% of full scale */
  for (size_t i = 1; i + 1 < test->n; i++) {
    float left = test->x[i] - test->x[i - 1];
    float right = test->x[i + 1] - test->x[i];
    /* Isolated spike: large excursions on both sides, reversing sign. */
    if (fabsf(left) > thresh && fabsf(right) > thresh &&
        ((left > 0) != (right > 0)) &&
        fabsf(test->x[i] - test->x[i - 1]) > thresh) {
      pops++;
    }
    if (test->x[i] >= 32767.0f || test->x[i] <= -32768.0f) clips++;
  }
  *pops_out = pops;
  *clips_out = clips;
}

/* Whole-signal time-aligned error for cross-rate comparison. */
static long find_best_lag(const Signal *ref, const Signal *test) {
  /* Cross-correlate over a +/- 120 sample window (generous group-delay
     margin) on the common time grid using linear interpolation. */
  long best_lag = 0;
  double best_corr = -1e300;
  long margin = 120;
  size_t lo = (size_t) (4800 > margin ? 4800 : margin);
  for (long d = -margin; d <= margin; d++) {
    double acc = 0, count = 0;
    for (size_t i = lo; i + lo < test->n; i += 2) {
      double t = ((double) i + d) / test->rate;
      if (t < 0) continue;
      float r = sample_at(ref, t);
      acc += (double) r * test->x[i];
      count++;
    }
    acc /= count > 0 ? count : 1;
    if (acc > best_corr) {
      best_corr = acc;
      best_lag = d;
    }
  }
  return best_lag;
}

static void relative_evaluate(const Signal *ref, const Signal *test,
                              double *snr_db, float *mos_out, long *lag_out) {
  long lag = find_best_lag(ref, test);
  if (lag_out) *lag_out = lag;
  double sig_e = 0, err_e = 0;
  size_t span = test->n;
  for (size_t i = 0; i < span; i++) {
    double t = ((double) i + lag) / test->rate;
    float r = sample_at(ref, t);
    float e = test->x[i] - r;
    sig_e += (double) r * r;
    err_e += (double) e * e;
  }
  *snr_db = 10.0 * log10((sig_e + 1e-9) / (err_e + 1e-9));
  *mos_out = relative_mos(sig_e, err_e);
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr,
            "usage: %s <ref> <ref_rate> <test> <test_rate> [label] "
            "[bandlimit_hz]\n",
            argv[0]);
    return 1;
  }
  Signal ref, test;
  if (load_pcm(argv[1], atoi(argv[2]), &ref) ||
      load_pcm(argv[3], atoi(argv[4]), &test)) {
    return 2;
  }
  const char *label = (argc >= 6) ? argv[5] : "test";
  double bandlimit = (argc >= 7) ? atof(argv[6]) : 0.0;
  Signal ref_eff = ref, test_eff = test;
  if (bandlimit > 0) {
    low_pass(&ref, bandlimit, &ref_eff);
    low_pass(&test, bandlimit, &test_eff);
  }
  float seg_snr, mos;
  long pops, clips;
  evaluate(&ref_eff, &test_eff, &seg_snr, &mos, &pops, &clips);
  double rel_snr;
  float rel_mos;
  long rel_lag;
  relative_evaluate(&ref_eff, &test_eff, &rel_snr, &rel_mos, &rel_lag);
  double dur_ref = (double) ref.n / ref.rate;
  double dur_test = (double) test.n / test.rate;

  printf("label            : %s\n", label);
  if (bandlimit > 0) printf("band limit       : %.0f Hz\n", bandlimit);
  printf("ref samples/rms  : %zu (%.2f s, rms %.1f)\n",
         ref.n, dur_ref, rms(&ref));
  printf("test samples/rms : %zu (%.3f s, rms %.1f)\n",
         test.n, dur_test, rms(&test));
  printf("duration delta   : %+.3f ms\n",
         (dur_test - dur_ref) * 1000.0);
  printf("segmental SNR    : %.2f dB\n", seg_snr);
  printf("MOS-like score   : %.3f\n", mos);
  printf("whole-sig SNR    : %.2f dB\n", rel_snr);
  printf("relative MOS     : %.3f\n", rel_mos);
  printf("alignment lag    : %ld samples\n", rel_lag);
  printf("pops/clicks      : %ld\n", pops);
  printf("hard clips       : %ld\n", clips);

  if (bandlimit > 0) {
    free(ref_eff.x);
    free(test_eff.x);
  }
  free(ref.x);
  free(test.x);
  return 0;
}
