/* Generates 500 deterministic noisy-speech samples (48 kHz mono, 16-bit WAV).
 *
 * "Speech" is a deterministic voiced/unvoiced formant source; noise covers
 * white, pink-ish, brown, babble-ish multi-tone, tone, chirp, impulse and
 * amplitude-modulated types. Mixing SNR, level and duration are varied per
 * sample. Every sample is reproducible from its index, and manifest.csv lists
 * the exact parameters of every file. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#include "qa_common.h"

#define BASE_SEED 0x524E4E31u  /* "RNN1" */

static int mkdir_p(const char *path) {
  char tmp[512];
  size_t len;
  size_t i;
  snprintf(tmp, sizeof(tmp), "%s", path);
  len = strlen(tmp);
  if (len == 0 || len >= sizeof(tmp)) return -1;
  for (i = 1; i < len; i++) {
    if (tmp[i] == '/') {
      tmp[i] = '\0';
      if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
      tmp[i] = '/';
    }
  }
  if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
  return 0;
}

enum {
  NOISE_WHITE = 0,
  NOISE_PINK,
  NOISE_BROWN,
  NOISE_BABBLE,
  NOISE_TONE,
  NOISE_CHIRP,
  NOISE_IMPULSE,
  NOISE_AM,
  NOISE_TYPES
};

static const char *noise_name(int t) {
  static const char *names[NOISE_TYPES] = {
    "white", "pink", "brown", "babble",
    "tone", "chirp", "impulse", "am_mod"};
  return names[t];
}

/* Three-formant voiced source + noise excitation, with pitch and amplitude
 * contours. Purely deterministic given the per-sample rng. */
static void gen_speech(float *x, size_t n, QaRng *rng) {
  size_t i;
  double f0 = 90.0 + 110.0 * qa_rng_unit(rng);          /* 90-200 Hz */
  double formant[3] = {
    700.0 + 200.0 * qa_rng_unit(rng),
    1100.0 + 400.0 * qa_rng_unit(rng),
    2400.0 + 600.0 * qa_rng_unit(rng)
  };
  double bw[3] = {80.0, 120.0, 160.0};
  double gain[3] = {1.0, 0.55, 0.28};
  double phase = 0.0;
  double hp[2] = {0.0, 0.0};
  double syllable_period = (0.18 + 0.22 * qa_rng_unit(rng)) * QA_SR;
  double syllable_phase = qa_rng_unit(rng) * syllable_period;
  double voiced_frac = 0.7 + 0.25 * qa_rng_unit(rng);

  for (i = 0; i < n; i++) {
    double t = (double)i / QA_SR;
    double contour = 1.0 + 0.12 * sin(2.0 * QA_PI * 1.8 * t) +
                      0.08 * sin(2.0 * QA_PI * 0.7 * t + 1.3);
    double pitch = f0 * contour;
    double syll;
    double env;
    double exc;
    double v;
    int k;
    phase += 2.0 * QA_PI * pitch / QA_SR;
    if (phase > 2.0 * QA_PI) phase -= 2.0 * QA_PI;

    syll = 0.5 + 0.5 * sin(2.0 * QA_PI * (i + syllable_phase) /
                           syllable_period);
    syll = syll * syll;
    env = 0.25 + 0.75 * syll;

    if (qa_rng_unit(rng) < voiced_frac) {
      exc = 0.6 * sin(phase) + 0.25 * sin(2.0 * phase) +
            0.12 * sin(3.0 * phase);
    } else {
      exc = 0.5 * qa_rng_gauss(rng);
    }

    v = 0.0;
    for (k = 0; k < 3; k++) {
      double r = exp(-QA_PI * bw[k] / QA_SR);
      double w = 2.0 * QA_PI * formant[k] / QA_SR;
      double b0 = (1.0 - r) * sin(w);
      /* cheap one-pole resonator on the excitation */
      double y = b0 * exc + 2.0 * r * cos(w) * hp[0] - r * r * hp[1];
      hp[1] = hp[0];
      hp[0] = y;
      v += gain[k] * y;
    }
    x[i] = (float)(env * v * 0.5);
  }
}

static void gen_noise(float *x, size_t n, int type, QaRng *rng) {
  size_t i;
  QaWhiteSrc white;
  QaLpSrc lp;
  double tone_freq;
  double tone_phase;
  double chirp_f0, chirp_f1, chirp_period;
  double am_freq;
  double next_impulse = 0.0;
  double pink[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double chirp_phase = 0.0;

  qa_white_init(&white, qa_rng_u64(rng));
  qa_lp_init(&lp, qa_rng_u64(rng), 0.85f);
  tone_freq = 300.0 + 2700.0 * qa_rng_unit(rng);
  tone_phase = 0.0;
  chirp_f0 = 200.0;
  chirp_f1 = 3000.0;
  chirp_period = (1.0 + 2.0 * qa_rng_unit(rng)) * QA_SR;
  am_freq = 2.0 + 8.0 * qa_rng_unit(rng);

  for (i = 0; i < n; i++) {
    double t = (double)i / QA_SR;
    double v = 0.0;
    switch (type) {
      case NOISE_WHITE:
        v = qa_white_next(&white);
        break;
      case NOISE_PINK: {
        /* Paul Kellet-style pink filter. */
        double w = qa_white_next(&white);
        pink[0] = 0.99886 * pink[0] + w * 0.0555179;
        pink[1] = 0.99332 * pink[1] + w * 0.0750759;
        pink[2] = 0.96900 * pink[2] + w * 0.1538520;
        pink[3] = 0.86650 * pink[3] + w * 0.3104856;
        pink[4] = 0.55000 * pink[4] + w * 0.5329522;
        pink[5] = -0.7616 * pink[5] - w * 0.0168980;
        v = (pink[0] + pink[1] + pink[2] + pink[3] + pink[4] + pink[5] +
             pink[6] + w * 0.5362) * 0.11;
        pink[6] = w * 0.115926;
        break;
      }
      case NOISE_BROWN:
        v = qa_lp_next(&lp) * 3.5;
        break;
      case NOISE_BABBLE: {
        int k;
        v = 0.0;
        for (k = 0; k < 6; k++) {
          double f = 120.0 * (k + 2) +
                     40.0 * sin(2.0 * QA_PI * (0.5 + 0.2 * k) * t + k);
          v += sin(2.0 * QA_PI * f * t + k * 1.7) / 6.0;
        }
        v += 0.3 * qa_white_next(&white);
        break;
      }
      case NOISE_TONE:
        tone_phase += 2.0 * QA_PI * tone_freq / QA_SR;
        v = 0.8 * sin(tone_phase) + 0.1 * qa_white_next(&white);
        break;
      case NOISE_CHIRP: {
        double phase_pos = fmod((double)i, chirp_period) / chirp_period;
        double freq = chirp_f0 + (chirp_f1 - chirp_f0) * phase_pos;
        chirp_phase += 2.0 * QA_PI * freq / QA_SR;
        if (chirp_phase > 2.0 * QA_PI) chirp_phase -= 2.0 * QA_PI;
        v = 0.7 * sin(chirp_phase) + 0.1 * qa_white_next(&white);
        break;
      }
      case NOISE_IMPULSE:
        if ((double)i >= next_impulse) {
          v = (qa_rng_signed(rng)) * 3.0;
          next_impulse = i + QA_SR * (0.05 + 0.45 * qa_rng_unit(rng));
        } else {
          v = 0.05 * qa_white_next(&white);
        }
        break;
      case NOISE_AM:
        v = qa_white_next(&white) *
            (0.35 + 0.65 * (0.5 + 0.5 * sin(2.0 * QA_PI * am_freq * t)));
        break;
    }
    x[i] = (float)v;
  }
}

static void scale_to_rms(float *x, size_t n, float target_rms) {
  float r = qa_rms(x, n);
  float g;
  size_t i;
  if (r < 1e-9f) return;
  g = target_rms / r;
  for (i = 0; i < n; i++) x[i] *= g;
}

static void apply_fade_edges(float *x, size_t n, size_t fade) {
  size_t i;
  if (fade > n / 2) fade = n / 2;
  for (i = 0; i < fade; i++) {
    float g = 0.5f - 0.5f * cosf(QA_PI * (float)i / (float)fade);
    x[i] *= g;
    x[n - 1 - i] *= g;
  }
}

int main(int argc, char **argv) {
  const char *out_dir = (argc > 1) ? argv[1] : "tests/qa/dataset";
  const char *manifest_path = (argc > 2) ? argv[2]
                                         : "tests/qa/results/manifest.csv";
  FILE *manifest;
  int idx;

  if (mkdir_p(out_dir) != 0) {
    fprintf(stderr, "cannot create %s\n", out_dir);
    return 1;
  }
  manifest = fopen(manifest_path, "w");
  if (manifest == NULL) {
    fprintf(stderr, "cannot write %s\n", manifest_path);
    return 1;
  }
  fprintf(manifest,
          "index,file,duration_s,frames,noise_type,snr_db,speech_rms,"
          "noise_rms,peak,seed\n");

  for (idx = 0; idx < QA_N_SAMPLES; idx++) {
    QaRng rng;
    int frames;
    size_t n;
    float *speech, *noise, *mix;
    int noise_type;
    float snr_db, speech_rms, noise_rms, peak;
    char path[512];
    size_t i;

    qa_rng_seed(&rng, ((uint64_t)BASE_SEED << 32) ^ (uint64_t)idx *
                          0x9e3779b97f4a7c15ULL);
    frames = 100 + (int)(qa_rng_u64(&rng) % 201);       /* 1.0-3.0 s */
    n = (size_t)frames * QA_FRAME;
    noise_type = (int)(qa_rng_u64(&rng) % NOISE_TYPES);
    snr_db = -10.0f + (float)(qa_rng_u64(&rng) % 251) / 10.0f; /* -10..15 */
    speech_rms = 0.04f + 0.12f * qa_rng_unit(&rng);

    speech = (float *)calloc(n, sizeof(float));
    noise = (float *)calloc(n, sizeof(float));
    mix = (float *)calloc(n, sizeof(float));
    if (speech == NULL || noise == NULL || mix == NULL) return 2;

    gen_speech(speech, n, &rng);
    gen_noise(noise, n, noise_type, &rng);
    scale_to_rms(speech, n, speech_rms);
    noise_rms = speech_rms / powf(10.0f, snr_db / 20.0f);
    scale_to_rms(noise, n, noise_rms);

    peak = 0.0f;
    for (i = 0; i < n; i++) {
      mix[i] = speech[i] + noise[i];
      if (fabsf(mix[i]) > peak) peak = fabsf(mix[i]);
    }
    /* Keep headroom; clamp the rare mix that exceeds +/-1. */
    if (peak > 0.95f) {
      float g = 0.95f / peak;
      for (i = 0; i < n; i++) mix[i] *= g;
      peak = 0.95f;
    }
    apply_fade_edges(mix, n, QA_FRAME);

    snprintf(path, sizeof(path), "%s/sample_%03d.wav", out_dir, idx);
    if (qa_wav_write(path, mix, n) != 0) {
      fprintf(stderr, "write failed: %s\n", path);
      return 3;
    }
    fprintf(manifest,
            "%d,sample_%03d.wav,%.3f,%d,%s,%.1f,%.5f,%.5f,%.4f,%llu\n",
            idx, idx, (double)n / QA_SR, frames, noise_name(noise_type),
            snr_db, speech_rms, noise_rms, peak,
            (unsigned long long)rng.state);

    free(speech);
    free(noise);
    free(mix);
    if ((idx + 1) % 50 == 0) fprintf(stderr, "generated %d/%d\n", idx + 1,
                                     QA_N_SAMPLES);
  }
  fclose(manifest);
  fprintf(stderr, "dataset: %d files in %s\n", QA_N_SAMPLES, out_dir);
  return 0;
}
