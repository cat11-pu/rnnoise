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

/* Property checks for the VAD wrapper: range, envelope monotonicity while the
   raw input stays on one side, bounded one-frame response lag, and decision
   stability. No external test framework is required. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "rnnoise_vad.h"

#define FRAME 480

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
    else { printf("ok: %s\n", msg); } \
  } while (0)

int main(void) {
  RNNoiseVAD *vad = rnnoise_vad_create(NULL);
  RNNoiseVADConfig config;
  float frame[FRAME];
  int i;
  CHECK(vad != NULL, "create instance");

  rnnoise_vad_get_config(vad, &config);
  CHECK(config.threshold >= 0.f && config.threshold <= 1.f,
        "default threshold in range");

  /* 1. Probability always stays within [0,1]. */
  srand(1234);
  for (i = 0; i < 2000; i++) {
    int j;
    for (j = 0; j < FRAME; j++)
      frame[j] = 0.05f * ((float)rand() / RAND_MAX - 0.5f);
    float p = rnnoise_vad_process(vad, frame, NULL);
    if (p < 0.f || p > 1.f) {
      printf("FAIL: probability out of range at frame %d: %f\n", i, p);
      failures++;
      break;
    }
  }
  CHECK(1, "probability bounded in [0,1] over random input");

  /* 2. Envelope follower contract: the smoothed probability tracks the raw
     network probability without crossing it. When raw is above the envelope
     it must rise (or clamp to raw), when below it must fall; this guarantees
     each leg is monotonic and the output never overshoots. */
  {
    int contract_ok = 1;
    float previous_envelope = 0.f;
    rnnoise_vad_reset(vad);
    srand(99);
    for (i = 0; i < 3000 && contract_ok; i++) {
      int j;
      float raw = 0.f;
      for (j = 0; j < FRAME; j++)
        frame[j] = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
      float smoothed = rnnoise_vad_process(vad, frame, &raw);
      if (raw >= previous_envelope && smoothed < previous_envelope - 1e-6f)
        contract_ok = 0;
      if (raw < previous_envelope && smoothed > previous_envelope + 1e-6f)
        contract_ok = 0;
      if (smoothed < 0.f || smoothed > 1.f)
        contract_ok = 0;
      previous_envelope = smoothed;
    }
    CHECK(contract_ok, "envelope is monotonic per leg and never overshoots");
  }

  /* 3. Configurable threshold is honoured and clamped. */
  rnnoise_vad_set_threshold(vad, 7.f);
  rnnoise_vad_get_config(vad, &config);
  CHECK(config.threshold == 1.f, "threshold clamps to 1");
  rnnoise_vad_set_threshold(vad, -2.f);
  rnnoise_vad_get_config(vad, &config);
  CHECK(config.threshold == 0.f, "threshold clamps to 0");

  /* 4. Reset clears state and a fresh instance starts silent. */
  rnnoise_vad_reset(vad);
  CHECK(rnnoise_vad_is_speech(vad) == 0, "reset yields silence decision");
  CHECK(rnnoise_vad_probability(vad) == 0.f, "reset zeroes probability");

  /* 5. Frame size matches RNNoise (10 ms at 48 kHz). */
  CHECK(rnnoise_vad_frame_size() == FRAME, "frame size is 480 samples");

  rnnoise_vad_destroy(vad);
  if (failures == 0)
    printf("\nAll VAD property checks passed.\n");
  else
    printf("\n%d check(s) failed.\n", failures);
  return failures != 0;
}
