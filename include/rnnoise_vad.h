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

#ifndef RNNOISE_VAD_H
#define RNNOISE_VAD_H

#include "rnnoise.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file rnnoise_vad.h
 *
 * Voice activity detection on top of RNNoise.
 *
 * The wrapper turns the per-frame raw network probability into a
 * monotonically behaving speech probability and a stable binary decision
 * using:
 *  - a causal attack/release envelope follower (no overshoot, no flicker);
 *  - hysteresis around a configurable threshold (boundary stability);
 *  - minimum speech / silence durations (debounced state transitions).
 *
 * All processing is causal and frame synchronous: each call consumes one
 * RNNoise frame (rnnoise_get_frame_size(), 10 ms at 48 kHz) and returns the
 * probability and decision for that same frame. The algorithmic latency is
 * therefore at most one frame. The probability is a causal envelope follower
 * (monotonic on each rise/fall leg, never overshooting the network output);
 * the binary decision adds hysteresis and a speech hold (hangover) so brief
 * intra-speech dips do not fragment a speech segment.
 */

typedef struct RNNoiseVAD RNNoiseVAD;

/** Tunable VAD parameters. */
typedef struct RNNoiseVADConfig {
  /** Speech decision threshold in [0,1]. Default 0.45. */
  float threshold;
  /** Hysteresis margin in [0, threshold]. Default 0.10.
   *  Speech turns off below threshold-hysteresis while speech is active. */
  float hysteresis;
  /** Attack smoothing coefficient in [0,1]. Default 0.5.
   *  Larger = slower rise; the envelope never overshoots the raw input. */
  float attack;
  /** Release smoothing coefficient in [0,1]. Default 0.4.
   *  Larger = slower fall; the envelope never undershoots the raw input. */
  float release;
  /** Minimum consecutive speech frames before speech is reported. Default 2. */
  int min_speech_frames;
  /** Minimum consecutive silence frames before silence is reported. Default 4. */
  int min_silence_frames;
  /** Speech hold (hangover) in frames: once speech is active, tolerate up to
   *  this many consecutive below-threshold frames before switching to
   *  silence, suppressing gaps inside an utterance. Default 3. */
  int speech_hold_frames;
} RNNoiseVADConfig;

/** Fill a config with the default values. */
RNNOISE_EXPORT void rnnoise_vad_config_init(RNNoiseVADConfig *config);

/** Allocate a VAD instance bound to an RNNoise model (NULL = default).
 *  The instance owns an internal DenoiseState and must be released with
 *  rnnoise_vad_destroy(). A custom model must outlive the instance. */
RNNOISE_EXPORT RNNoiseVAD *rnnoise_vad_create(RNNModel *model);

/** Allocate a VAD instance wrapping an externally owned DenoiseState.
 *  The DenoiseState is not freed by rnnoise_vad_destroy(). */
RNNOISE_EXPORT RNNoiseVAD *rnnoise_vad_create_with_state(DenoiseState *st);

/** Free a VAD instance. */
RNNOISE_EXPORT void rnnoise_vad_destroy(RNNoiseVAD *vad);

/** Apply a full parameter set. Out-of-range values are clamped. */
RNNOISE_EXPORT int rnnoise_vad_set_config(RNNoiseVAD *vad, const RNNoiseVADConfig *config);

/** Copy the active parameter set. */
RNNOISE_EXPORT void rnnoise_vad_get_config(RNNoiseVAD *vad, RNNoiseVADConfig *config);

/** Update only the decision threshold (clamped to [0,1]). */
RNNOISE_EXPORT int rnnoise_vad_set_threshold(RNNoiseVAD *vad, float threshold);

/** Reset all internal state while keeping the current parameters. */
RNNOISE_EXPORT void rnnoise_vad_reset(RNNoiseVAD *vad);

/**
 * Process one frame and return the VAD result.
 *
 * @param vad      VAD instance.
 * @param frame    rnnoise_get_frame_size() input samples (float PCM).
 * @param raw_prob Optional output for the unsmoothed network probability,
 *                 may be NULL.
 * @return Smoothed speech probability in [0,1]; query the binary decision
 *         with rnnoise_vad_is_speech().
 */
RNNOISE_EXPORT float rnnoise_vad_process(RNNoiseVAD *vad, const float *frame, float *raw_prob);

/** Binary speech decision for the frame passed to the last process call. */
RNNOISE_EXPORT int rnnoise_vad_is_speech(const RNNoiseVAD *vad);

/** Smoothed speech probability for the last processed frame. */
RNNOISE_EXPORT float rnnoise_vad_probability(const RNNoiseVAD *vad);

/** Number of samples consumed per call (10 ms at 48 kHz). */
RNNOISE_EXPORT int rnnoise_vad_frame_size(void);

#ifdef __cplusplus
}
#endif

#endif
