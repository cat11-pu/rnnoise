#ifndef QA_COMMON_H
#define QA_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#define QA_SR 48000
#define QA_FRAME 480
#define QA_N_SAMPLES 500
#define QA_PI 3.14159265358979323846

/* Deterministic 64-bit PRNG (SplitMix64). Same seed -> identical samples on
 * every platform, so the 500-file test set is fully reproducible. */
typedef struct {
  uint64_t state;
} QaRng;

static inline void qa_rng_seed(QaRng *r, uint64_t seed) {
  r->state = seed;
}

static inline uint64_t qa_rng_u64(QaRng *r) {
  uint64_t z = (r->state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

/* Uniform float in [-1, 1). */
static inline float qa_rng_signed(QaRng *r) {
  return (float)((int64_t)qa_rng_u64(r)) / (float)(1ULL << 63);
}

/* Uniform float in [0, 1). */
static inline float qa_rng_unit(QaRng *r) {
  return (float)(qa_rng_u64(r) >> 11) / (float)(1ULL << 53);
}

/* Gaussian via Box-Muller, mean 0, variance 1. */
static inline float qa_rng_gauss(QaRng *r) {
  float u1 = qa_rng_unit(r);
  float u2 = qa_rng_unit(r);
  if (u1 < 1e-12f) u1 = 1e-12f;
  return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * QA_PI * u2);
}

/* Gaussian white-noise source with persistent state. */
typedef struct {
  QaRng rng;
} QaWhiteSrc;

static inline void qa_white_init(QaWhiteSrc *s, uint64_t seed) {
  qa_rng_seed(&s->rng, seed);
}

static inline float qa_white_next(QaWhiteSrc *s) {
  return qa_rng_gauss(&s->rng);
}

/* One-pole low-pass filtered noise (brown/pink-ish), stateful. */
typedef struct {
  QaWhiteSrc white;
  float coeff;
  float mem;
} QaLpSrc;

static inline void qa_lp_init(QaLpSrc *s, uint64_t seed, float coeff) {
  qa_white_init(&s->white, seed);
  s->coeff = coeff;
  s->mem = 0.0f;
}

static inline float qa_lp_next(QaLpSrc *s) {
  float x = qa_white_next(&s->white);
  s->mem = s->coeff * s->mem + (1.0f - s->coeff) * x;
  return s->mem;
}

/* Simple mono WAV (16-bit PCM) helpers. */
int qa_wav_write(const char *path, const float *x, size_t n);
int qa_wav_read(const char *path, float **x, size_t *n);

/* Streaming RNNoise wrapper: feed arbitrary chunk sizes, samples are
 * buffered internally and rnnoise_process_frame() is always called with
 * exactly QA_FRAME samples. Processed frames are queued in a ring buffer so
 * bursts larger than one frame are never overwritten. */
#define QA_STREAM_RING_FRAMES 64
typedef struct {
  void *st;
  float in_buf[QA_FRAME];
  float ring[QA_STREAM_RING_FRAMES * QA_FRAME];
  int in_fill;
  int ring_head;    /* next sample to read */
  int ring_count;   /* samples currently queued */
} QaStream;

void qa_stream_init(QaStream *s, void *st);
size_t qa_stream_process(QaStream *s, const float *in, size_t n,
                         float *out, size_t out_cap);

/* RMS of a buffer. */
float qa_rms(const float *x, size_t n);

/* Mean absolute delta (used as the local activity baseline for click tests). */
float qa_mean_abs_delta(const float *x, size_t n);

/* Cross-correlation lag of y vs x in samples, searched in [min_lag,max_lag].
 * Positive lag means y is delayed relative to x. */
long qa_estimate_lag(const float *x, const float *y, size_t n,
                     long min_lag, long max_lag);

/* Frame-energy envelope lag (10 ms frames). Robust to spectral shaping
 * because RNNoise preserves the temporal envelope. */
long qa_estimate_envelope_lag(const float *x, const float *y, size_t frames,
                              long min_lag_frames, long max_lag_frames);

/* Impulse-response lag in samples: feed one unit impulse through a fresh
 * state and locate the frame carrying the response energy. */
long qa_impulse_lag(void);

#endif
