/* Copyright (c) 2024 The RNNoise authors. */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rnnoise_vad.h"

/* Confusion counters over frames. */
typedef struct {
  long tp, tn, fp, fn;
} VadStats;

static void stats_update(VadStats *s, int truth, int decision) {
  if (truth && decision) s->tp++;
  else if (truth) s->fn++;
  else if (decision) s->fp++;
  else s->tn++;
}

static double safe_div(double num, double den) {
  return den > 0.0 ? num / den : 0.0;
}

static void print_stats(const char *title, const VadStats *s) {
  long speech = s->tp + s->fn;
  long silence = s->tn + s->fp;
  long total = speech + silence;
  double false_alarm = safe_div(s->fp, silence);
  double miss = safe_div(s->fn, speech);
  printf("%s\n", title);
  printf("  frames total=%ld speech=%ld silence=%ld\n", total, speech, silence);
  printf("  TP=%ld TN=%ld FP=%ld FN=%ld\n", s->tp, s->tn, s->fp, s->fn);
  printf("  false_alarm_rate=%.4f%%  miss_rate=%.4f%%\n",
         100.0 * false_alarm, 100.0 * miss);
  printf("  sensitivity=%.4f%%  specificity=%.4f%%  accuracy=%.4f%%\n",
         100.0 * safe_div(s->tp, speech),
         100.0 * safe_div(s->tn, silence),
         100.0 * safe_div((double)(s->tp + s->tn), (double)total));
}

static void usage(const char *prog) {
  fprintf(stderr,
    "usage:\n"
    "  %s -p <input.pcm> [output.csv] [threshold]\n"
    "      print frame index, raw prob, smoothed prob, decision (CSV)\n"
    "  %s -e <input.pcm> <labels.txt> [threshold]\n"
    "      score VAD against per-frame labels (1=speech, 0=silence)\n"
    "  %s -s <input.pcm> <labels.txt>\n"
    "      sweep thresholds 0.05..0.95 and print error rates\n"
    "The first 10 frames are excluded from scoring as model warm-up.\n"
    "PCM: 48 kHz mono 16-bit signed, frames of %d samples\n",
    prog, prog, prog, rnnoise_vad_frame_size());
}

int main(int argc, char **argv) {
  int mode;
  const char *pcm_path;
  const char *aux_path = NULL;
  float threshold = -1.f;
  FILE *fpcm, *faux = NULL;
  RNNoiseVAD *vad;
  RNNoiseVADConfig config;
  float frame[480];
  short tmp[480];
  long index = 0;
  VadStats stats = {0};

  if (argc < 3) {
    usage(argv[0]);
    return 1;
  }
  if (strcmp(argv[1], "-p") == 0) mode = 'p';
  else if (strcmp(argv[1], "-e") == 0) mode = 'e';
  else if (strcmp(argv[1], "-s") == 0) mode = 's';
  else {
    usage(argv[0]);
    return 1;
  }
  pcm_path = argv[2];
  if (mode == 'p') {
    if (argc >= 4) aux_path = argv[3];
    if (argc >= 5) threshold = (float)atof(argv[4]);
  } else {
    if (argc < 4) {
      usage(argv[0]);
      return 1;
    }
    aux_path = argv[3];
    if (mode == 'e' && argc >= 5) threshold = (float)atof(argv[4]);
  }

  fpcm = fopen(pcm_path, "rb");
  if (fpcm == NULL) {
    perror(pcm_path);
    return 1;
  }
  if (aux_path != NULL && mode != 'p') {
    faux = fopen(aux_path, "r");
    if (faux == NULL) {
      perror(aux_path);
      return 1;
    }
  }

  vad = rnnoise_vad_create(NULL);
  rnnoise_vad_config_init(&config);
  if (mode != 's' && threshold >= 0.f) {
    rnnoise_vad_set_threshold(vad, threshold);
  }
  if (mode != 's') {
    rnnoise_vad_get_config(vad, &config);
  }

  if (mode == 'p') {
    FILE *fout = NULL;
    if (aux_path != NULL) {
      fout = fopen(aux_path, "w");
      if (fout == NULL) {
        perror(aux_path);
        return 1;
      }
    }
    if (fout != NULL)
      fprintf(fout, "frame,time_ms,raw_prob,smoothed_prob,speech\n");
    else
      printf("frame,time_ms,raw_prob,smoothed_prob,speech\n");
    while (fread(tmp, sizeof(short), 480, fpcm) == 480) {
      int i;
      float raw = 0.f;
      float prob;
      for (i = 0; i < 480; i++) frame[i] = (float)tmp[i];
      prob = rnnoise_vad_process(vad, frame, &raw);
      if (fout != NULL)
        fprintf(fout, "%ld,%ld,%.6f,%.6f,%d\n", index, index * 10, raw,
                prob, rnnoise_vad_is_speech(vad));
      else
        printf("%ld,%ld,%.6f,%.6f,%d\n", index, index * 10, raw, prob,
               rnnoise_vad_is_speech(vad));
      index++;
    }
    if (fout != NULL) fclose(fout);
  } else if (mode == 'e') {
    int truth;
    while (fread(tmp, sizeof(short), 480, fpcm) == 480 &&
           fscanf(faux, "%d", &truth) == 1) {
      int i;
      for (i = 0; i < 480; i++) frame[i] = (float)tmp[i];
      rnnoise_vad_process(vad, frame, NULL);
      if (index >= 10)
        stats_update(&stats, truth != 0, rnnoise_vad_is_speech(vad));
      index++;
    }
    printf("threshold=%.2f hysteresis=%.2f attack=%.2f release=%.2f "
           "min_speech=%d min_silence=%d\n",
           config.threshold, config.hysteresis, config.attack, config.release,
           config.min_speech_frames, config.min_silence_frames);
    print_stats("VAD frame-level statistics", &stats);
  } else {
    /* Sweep: collect raw probabilities and labels first, then evaluate each
       threshold with the same causal smoothing/debounce pipeline. */
    float *raw = NULL;
    int *labels = NULL;
    size_t cap = 0, n = 0;
    long n_frame = 0;
    int truth;
    while (fread(tmp, sizeof(short), 480, fpcm) == 480 &&
           fscanf(faux, "%d", &truth) == 1) {
      int i;
      float raw_prob = 0.f;
      if (n == cap) {
        cap = cap ? cap * 2 : 1024;
        raw = realloc(raw, cap * sizeof(*raw));
        labels = realloc(labels, cap * sizeof(*labels));
      }
      for (i = 0; i < 480; i++) frame[i] = (float)tmp[i];
      rnnoise_vad_process(vad, frame, &raw_prob);
      if (n_frame >= 10) {
        raw[n] = raw_prob;
        labels[n] = truth != 0;
        n++;
      }
      n_frame++;
    }
    printf("threshold,false_alarm_pct,miss_pct,FP,FN,TN,TP\n");
    {
      int k;
      for (k = 5; k <= 95; k += 5) {
        float thr = k / 100.f;
        float envelope = 0.f;
        int speech = 0, run = 0;
        VadStats s = {0};
        size_t j;
        for (j = 0; j < n; j++) {
          float prev = envelope;
          float r = raw[j];
          float alpha = (r >= prev) ? config.attack : config.release;
          int candidate;
          envelope = alpha * prev + (1.f - alpha) * r;
          if (r >= prev && envelope > r) envelope = r;
          if (r < prev && envelope < r) envelope = r;
          candidate = speech ? envelope >= thr - config.hysteresis
                             : envelope >= thr;
          if (candidate == speech) {
            run = 0;
          } else {
            run++;
            if (candidate) {
              if (run >= config.min_speech_frames) {
                speech = 1;
                run = 0;
              }
            } else if (run > config.speech_hold_frames &&
                       run >= config.min_silence_frames) {
              speech = 0;
              run = 0;
            }
          }
          stats_update(&s, labels[j], speech);
        }
        printf("%.2f,%.4f,%.4f,%ld,%ld,%ld,%ld\n", thr,
               100.0 * safe_div(s.fp, s.tn + s.fp),
               100.0 * safe_div(s.fn, s.tp + s.fn),
               s.fp, s.fn, s.tn, s.tp);
      }
    }
    free(raw);
    free(labels);
  }

  rnnoise_vad_destroy(vad);
  fclose(fpcm);
  if (faux != NULL) fclose(faux);
  return 0;
}
