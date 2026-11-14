#include "resample.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define RATIO_MAX 4096

struct Resampler {
  int up;                    /* numerator of Fs_out/Fs_in, reduced       */
  int down;                  /* denominator of Fs_out/Fs_in, reduced     */
  int half_width;            /* sinc support, input samples each side    */
  double kaiser_beta;
  double fc;                 /* low-pass cutoff, fraction of input Nyq   */
  double i0_beta;

  float *ring;               /* circular history of input samples        */
  size_t ring_mask;          /* capacity - 1; capacity is a power of two  */
  long long newest;          /* global index of the last pushed sample    */
  long long count;           /* total samples pushed so far               */

  long long time_out;        /* next output sample index to produce      */
};

static int gcd_int(int a, int b) {
  while (b) { int t = a % b; a = b; b = t; }
  return a;
}

static double bessel_i0(double x) {
  double term = 1.0, sum = 1.0;
  double y = (x * x) / 4.0;
  for (int m = 1; m < 64; m++) {
    term *= y / ((double) m * m);
    sum += term;
    if (term < 1e-18 * sum) break;
  }
  return sum;
}

Resampler *resampler_create(int rate_in, int rate_out,
                            int half_width, double kaiser_beta) {
  if (rate_in <= 0 || rate_out <= 0 || half_width < 4) return NULL;
  int g = gcd_int(rate_in, rate_out);
  if (rate_in / g > RATIO_MAX || rate_out / g > RATIO_MAX) return NULL;

  Resampler *r = (Resampler *) calloc(1, sizeof(*r));
  if (!r) return NULL;
  r->up = rate_out / g;
  r->down = rate_in / g;
  r->half_width = half_width;
  r->kaiser_beta = kaiser_beta;
  double ratio = (double) r->up / r->down;
  r->fc = (ratio < 1.0 ? ratio : 1.0) * 0.985;
  r->i0_beta = bessel_i0(kaiser_beta);
  r->newest = -1;
  r->count = 0;
  r->time_out = 0;

  /* Ring must hold 2*half_width+2 taps; round up to a power of two. */
  size_t need = (size_t) 2 * half_width + 2;
  size_t cap = 8;
  while (cap < need) cap <<= 1;
  r->ring_mask = cap - 1;
  r->ring = (float *) calloc(cap, sizeof(float));
  if (!r->ring) { free(r); return NULL; }
  return r;
}

void resampler_destroy(Resampler *r) {
  if (!r) return;
  free(r->ring);
  free(r);
}

size_t resampler_max_out(const Resampler *r, size_t in_n) {
  return ((in_n + (size_t) r->half_width + 2) * (size_t) r->up) /
             (size_t) r->down + 2;
}

double resampler_delay_out(const Resampler *r) {
  return (double) r->half_width * r->up / r->down;
}

static inline double kaiser_window(const Resampler *r, double t) {
  double x = t / r->half_width;
  double arg = r->kaiser_beta * sqrt(1.0 - x * x);
  return bessel_i0(arg) / r->i0_beta;
}

static inline float ring_get(const Resampler *r, long long index) {
  /* Indices older than the ring window or not yet pushed read as zero,
     giving correct zero-padded transients at both stream boundaries. */
  if (index < 0 || index > r->newest) return 0.0f;
  if (r->newest - index > (long long) r->ring_mask) return 0.0f;
  return r->ring[(size_t) index & r->ring_mask];
}

static float interpolate(const Resampler *r, double center) {
  long long n0 = (long long) floor(center - 1e-12);
  int hw = r->half_width;
  double acc = 0.0;
  for (long long n = n0 - hw + 1; n <= n0 + hw; n++) {
    double local = (double) n - center;
    if (fabs(local) > hw) continue;
    double w = kaiser_window(r, local);
    double x = M_PI * r->fc * local;
    double sinc = (fabs(local) < 1e-12) ? 1.0 : sin(x) / x;
    acc += (double) ring_get(r, n) * r->fc * w * sinc;
  }
  return (float) acc;
}

static void emit(Resampler *r, float *out, size_t *out_n, size_t out_cap) {
  while (*out_n < out_cap) {
    double center = (double) r->time_out * r->down / r->up;
    /* Need the tap at floor(center)+half_width to be available. */
    long long need_index = (long long) floor(center + 1e-9) +
                           r->half_width;
    if (need_index > r->newest) break;
    out[(*out_n)++] = interpolate(r, center);
    r->time_out++;
  }
}

void resampler_process(Resampler *r, const float *in, size_t in_n,
                       float *out, size_t *out_n, size_t out_cap) {
  for (size_t i = 0; i < in_n; i++) {
    long long index = r->count++;
    r->ring[(size_t) index & r->ring_mask] = in[i];
    r->newest = index;
    emit(r, out, out_n, out_cap);
  }
}

void resampler_flush(Resampler *r, float *out, size_t *out_n,
                     size_t out_cap) {
  /* Zero-pad: advance the output clock until its rightmost tap would lie
     beyond the last real sample, emitting the decaying filter tail. */
  long long final_need = r->newest + r->half_width;
  size_t guard = 0;
  size_t max_extra = resampler_max_out(r, (size_t) (2 * r->half_width + 2));
  while (*out_n < out_cap && guard++ <= max_extra) {
    double center = (double) r->time_out * r->down / r->up;
    long long need_index = (long long) floor(center + 1e-9) +
                           r->half_width;
    if (need_index > final_need) break;
    out[(*out_n)++] = interpolate(r, center);
    r->time_out++;
  }
}
