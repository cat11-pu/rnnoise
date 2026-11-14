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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include "rnnoise_vad.h"

#define RNNOISE_VAD_BUILD
#include "denoise.h"

struct RNNoiseVAD {
  DenoiseState *st;
  int owns_state;
  RNNoiseVADConfig config;
  float envelope;
  float last_prob;
  int speech;
  int run;
};

static float clamp01(float x) {
  if (x < 0.f) return 0.f;
  if (x > 1.f) return 1.f;
  return x;
}

void rnnoise_vad_config_init(RNNoiseVADConfig *config) {
  config->threshold = 0.45f;
  config->hysteresis = 0.10f;
  config->attack = 0.5f;
  config->release = 0.4f;
  config->min_speech_frames = 2;
  config->min_silence_frames = 4;
  config->speech_hold_frames = 3;
}

static void sanitize_config(RNNoiseVADConfig *config) {
  config->threshold = clamp01(config->threshold);
  if (config->hysteresis < 0.f) config->hysteresis = 0.f;
  if (config->hysteresis > config->threshold) config->hysteresis = config->threshold;
  config->attack = clamp01(config->attack);
  config->release = clamp01(config->release);
  if (config->min_speech_frames < 1) config->min_speech_frames = 1;
  if (config->min_silence_frames < 1) config->min_silence_frames = 1;
  if (config->speech_hold_frames < 0) config->speech_hold_frames = 0;
}

RNNoiseVAD *rnnoise_vad_create_with_state(DenoiseState *st) {
  RNNoiseVAD *vad;
  if (st == NULL) return NULL;
  vad = (RNNoiseVAD *)calloc(1, sizeof(*vad));
  if (vad == NULL) return NULL;
  vad->st = st;
  vad->owns_state = 0;
  rnnoise_vad_config_init(&vad->config);
  rnnoise_vad_reset(vad);
  return vad;
}

RNNoiseVAD *rnnoise_vad_create(RNNModel *model) {
  RNNoiseVAD *vad;
  DenoiseState *st = rnnoise_create(model);
  if (st == NULL) return NULL;
  vad = rnnoise_vad_create_with_state(st);
  if (vad == NULL) {
    rnnoise_destroy(st);
    return NULL;
  }
  vad->owns_state = 1;
  return vad;
}

void rnnoise_vad_destroy(RNNoiseVAD *vad) {
  if (vad == NULL) return;
  if (vad->owns_state) rnnoise_destroy(vad->st);
  free(vad);
}

int rnnoise_vad_set_config(RNNoiseVAD *vad, const RNNoiseVADConfig *config) {
  if (vad == NULL || config == NULL) return -1;
  vad->config = *config;
  sanitize_config(&vad->config);
  return 0;
}

void rnnoise_vad_get_config(RNNoiseVAD *vad, RNNoiseVADConfig *config) {
  if (vad == NULL || config == NULL) return;
  *config = vad->config;
}

int rnnoise_vad_set_threshold(RNNoiseVAD *vad, float threshold) {
  if (vad == NULL) return -1;
  vad->config.threshold = clamp01(threshold);
  if (vad->config.hysteresis > vad->config.threshold)
    vad->config.hysteresis = vad->config.threshold;
  return 0;
}

void rnnoise_vad_reset(RNNoiseVAD *vad) {
  if (vad == NULL) return;
  vad->envelope = 0.f;
  vad->last_prob = 0.f;
  vad->speech = 0;
  vad->run = 0;
}

float rnnoise_vad_process(RNNoiseVAD *vad, const float *frame, float *raw_prob) {
  float frame_out[FRAME_SIZE];
  float raw;
  float alpha;
  float off_threshold;
  int candidate;
  if (vad == NULL || frame == NULL) return 0.f;

  raw = clamp01(rnnoise_process_frame(vad->st, frame_out, frame));
  if (raw_prob != NULL) *raw_prob = raw;

  /* Causal envelope follower: rises/falls toward the raw probability without
     ever crossing it, so each leg of the output is monotonic. Fast defaults
     keep the lag within one frame for steady-state crossings. */
  {
    float prev = vad->envelope;
    alpha = (raw >= prev) ? vad->config.attack : vad->config.release;
    vad->envelope = alpha * prev + (1.f - alpha) * raw;
    if (raw >= prev && vad->envelope > raw) vad->envelope = raw;
    if (raw < prev && vad->envelope < raw) vad->envelope = raw;
    vad->envelope = clamp01(vad->envelope);
  }
  vad->last_prob = vad->envelope;

  /* Hysteresis decision, debounced by minimum run lengths, plus a speech
     hold so short sub-threshold dips inside an utterance keep speech. */
  off_threshold = vad->config.threshold - vad->config.hysteresis;
  if (vad->speech)
    candidate = vad->envelope >= off_threshold;
  else
    candidate = vad->envelope >= vad->config.threshold;

  if (candidate == vad->speech) {
    vad->run = 0;
  } else if (candidate) {
    vad->run++;
    if (vad->run >= vad->config.min_speech_frames) {
      vad->speech = 1;
      vad->run = 0;
    }
  } else {
    vad->run++;
    if (vad->run > vad->config.speech_hold_frames &&
        vad->run >= vad->config.min_silence_frames) {
      vad->speech = 0;
      vad->run = 0;
    }
  }
  return vad->envelope;
}

int rnnoise_vad_is_speech(const RNNoiseVAD *vad) {
  return vad != NULL && vad->speech;
}

float rnnoise_vad_probability(const RNNoiseVAD *vad) {
  return vad != NULL ? vad->last_prob : 0.f;
}

int rnnoise_vad_frame_size(void) {
  return FRAME_SIZE;
}
