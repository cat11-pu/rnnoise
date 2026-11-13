#include "qa_common.h"

#include <stdlib.h>
#include <string.h>

#include "rnnoise.h"

static void put_u16(FILE *f, uint16_t v) {
  unsigned char b[2];
  b[0] = (unsigned char)(v & 0xff);
  b[1] = (unsigned char)((v >> 8) & 0xff);
  fwrite(b, 1, 2, f);
}

static void put_u32(FILE *f, uint32_t v) {
  unsigned char b[4];
  b[0] = (unsigned char)(v & 0xff);
  b[1] = (unsigned char)((v >> 8) & 0xff);
  b[2] = (unsigned char)((v >> 16) & 0xff);
  b[3] = (unsigned char)((v >> 24) & 0xff);
  fwrite(b, 1, 4, f);
}

static uint16_t get_u16(const unsigned char *b) {
  return (uint16_t)(b[0] | (b[1] << 8));
}

static uint32_t get_u32(const unsigned char *b) {
  return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
         ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

int qa_wav_write(const char *path, const float *x, size_t n) {
  FILE *f = fopen(path, "wb");
  size_t i;
  uint32_t data_bytes = (uint32_t)(n * 2);
  if (f == NULL) return -1;
  fwrite("RIFF", 1, 4, f);
  put_u32(f, 36 + data_bytes);
  fwrite("WAVE", 1, 4, f);
  fwrite("fmt ", 1, 4, f);
  put_u32(f, 16);
  put_u16(f, 1);                 /* PCM */
  put_u16(f, 1);                 /* mono */
  put_u32(f, QA_SR);
  put_u32(f, QA_SR * 2);         /* byte rate */
  put_u16(f, 2);                 /* block align */
  put_u16(f, 16);                /* bits per sample */
  fwrite("data", 1, 4, f);
  put_u32(f, data_bytes);
  for (i = 0; i < n; i++) {
    float v = x[i];
    int s;
    if (v > 1.0f) v = 1.0f;
    if (v < -1.0f) v = -1.0f;
    s = (int)lrintf(v * 32767.0f);
    put_u16(f, (uint16_t)(int16_t)s);
  }
  fclose(f);
  return 0;
}

int qa_wav_read(const char *path, float **out, size_t *out_n) {
  FILE *f = fopen(path, "rb");
  unsigned char hdr[44];
  size_t n;
  size_t got;
  uint16_t channels, bits;
  uint32_t rate, data_bytes;
  float *x;
  size_t i;
  if (f == NULL) return -1;
  if (fread(hdr, 1, 44, f) != 44 || memcmp(hdr, "RIFF", 4) != 0 ||
      memcmp(hdr + 8, "WAVE", 4) != 0 || memcmp(hdr + 12, "fmt ", 4) != 0 ||
      memcmp(hdr + 36, "data", 4) != 0) {
    fclose(f);
    return -2;
  }
  channels = get_u16(hdr + 22);
  rate = get_u32(hdr + 24);
  bits = get_u16(hdr + 34);
  data_bytes = get_u32(hdr + 40);
  if (channels != 1 || rate != QA_SR || bits != 16 ||
      data_bytes % 2 != 0) {
    fclose(f);
    return -3;
  }
  n = data_bytes / 2;
  x = (float *)malloc(n * sizeof(float));
  if (x == NULL) {
    fclose(f);
    return -4;
  }
  {
    unsigned char *raw = (unsigned char *)malloc(data_bytes);
    if (raw == NULL) {
      free(x);
      fclose(f);
      return -4;
    }
    got = fread(raw, 1, data_bytes, f);
    fclose(f);
    if (got != data_bytes) {
      free(raw);
      free(x);
      return -5;
    }
    for (i = 0; i < n; i++) {
      int16_t s = (int16_t)get_u16(raw + 2 * i);
      x[i] = (float)s / 32768.0f;
    }
    free(raw);
  }
  *out = x;
  *out_n = n;
  return 0;
}

void qa_stream_init(QaStream *s, void *st) {
  s->st = st;
  s->in_fill = 0;
  s->ring_head = 0;
  s->ring_count = 0;
}

size_t qa_stream_process(QaStream *s, const float *in, size_t n,
                         float *out, size_t out_cap) {
  size_t i;
  size_t written = 0;
  float frame_out[QA_FRAME];
  for (i = 0; i < n; i++) {
    s->in_buf[s->in_fill++] = in[i];
    if (s->in_fill == QA_FRAME) {
      int tail;
      if (s->ring_count + QA_FRAME > QA_STREAM_RING_FRAMES * QA_FRAME) break;
      rnnoise_process_frame((DenoiseState *)s->st, frame_out, s->in_buf);
      tail = (s->ring_head + s->ring_count) %
             (QA_STREAM_RING_FRAMES * QA_FRAME);
      if (tail + QA_FRAME <= QA_STREAM_RING_FRAMES * QA_FRAME) {
        memcpy(s->ring + tail, frame_out, QA_FRAME * sizeof(float));
      } else {
        int first = QA_STREAM_RING_FRAMES * QA_FRAME - tail;
        memcpy(s->ring + tail, frame_out, first * sizeof(float));
        memcpy(s->ring, frame_out + first,
               (QA_FRAME - first) * sizeof(float));
      }
      s->ring_count += QA_FRAME;
      s->in_fill = 0;
    }
    while (s->ring_count > 0 && written < out_cap) {
      out[written++] = s->ring[s->ring_head];
      s->ring_head = (s->ring_head + 1) %
                     (QA_STREAM_RING_FRAMES * QA_FRAME);
      s->ring_count--;
    }
  }
  return written;
}

float qa_rms(const float *x, size_t n) {
  double acc = 0.0;
  size_t i;
  if (n == 0) return 0.0f;
  for (i = 0; i < n; i++) acc += (double)x[i] * x[i];
  return (float)sqrt(acc / (double)n);
}

float qa_mean_abs_delta(const float *x, size_t n) {
  double acc = 0.0;
  size_t i;
  if (n < 2) return 0.0f;
  for (i = 1; i < n; i++) {
    double d = (double)x[i] - x[i - 1];
    acc += fabs(d);
  }
  return (float)(acc / (double)(n - 1));
}

long qa_estimate_lag(const float *x, const float *y, size_t n,
                     long min_lag, long max_lag) {
  long lag;
  long best_lag = min_lag;
  double best = -1e300;
  for (lag = min_lag; lag <= max_lag; lag++) {
    double acc = 0.0;
    size_t i;
    size_t lo = (lag >= 0) ? (size_t)lag : 0;
    size_t hi = (lag >= 0) ? n : (size_t)((long)n + lag);
    for (i = lo; i < hi; i++) {
      acc += (double)x[i] * (double)y[i - (size_t)lag];
    }
    if (acc > best) {
      best = acc;
      best_lag = lag;
    }
  }
  return best_lag;
}

long qa_estimate_envelope_lag(const float *x, const float *y, size_t frames,
                              long min_lag_frames, long max_lag_frames) {
  double *ex, *ey;
  size_t f;
  long lag, best_lag = min_lag_frames;
  double best = -1e300;
  ex = (double *)calloc(frames, sizeof(double));
  ey = (double *)calloc(frames, sizeof(double));
  if (ex == NULL || ey == NULL) {
    free(ex);
    free(ey);
    return -1;
  }
  for (f = 0; f < frames; f++) {
    size_t k;
    for (k = 0; k < QA_FRAME; k++) {
      ex[f] += (double)x[f * QA_FRAME + k] * x[f * QA_FRAME + k];
      ey[f] += (double)y[f * QA_FRAME + k] * y[f * QA_FRAME + k];
    }
  }
  /* Differentiate the energy envelopes so the alignment locks onto onsets
   * and energy changes rather than the dominant zero-lag baseline. */
  for (f = frames - 1; f > 0; f--) {
    ex[f] -= ex[f - 1];
    ey[f] -= ey[f - 1];
  }
  for (lag = min_lag_frames; lag <= max_lag_frames; lag++) {
    double acc = 0.0;
    size_t lo = (lag >= 0) ? (size_t)lag : 0;
    size_t hi = (lag >= 0) ? frames : (size_t)((long)frames + lag);
    size_t g;
    for (g = lo; g < hi; g++)
      acc += ex[g] * ey[g - (size_t)lag];
    if (acc > best) {
      best = acc;
      best_lag = lag;
    }
  }
  free(ex);
  free(ey);
  return best_lag;
}

long qa_impulse_lag(void) {
  DenoiseState *st = rnnoise_create(NULL);
  static const int probe_frames = 10;
  float in[QA_FRAME];
  float out[QA_FRAME];
  int f;
  long best_frame = -1;
  double best_energy = 0.0;
  if (st == NULL) return -1;
  for (f = 0; f < probe_frames; f++) {
    int i;
    double energy = 0.0;
    memset(in, 0, sizeof(in));
    if (f == 0) in[0] = 1.0f;
    rnnoise_process_frame(st, out, in);
    for (i = 0; i < QA_FRAME; i++) energy += (double)out[i] * out[i];
    if (energy > best_energy) {
      best_energy = energy;
      best_frame = f;
    }
  }
  rnnoise_destroy(st);
  return best_frame * QA_FRAME;
}
