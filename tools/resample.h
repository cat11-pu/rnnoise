/* Streaming polyphase rational-ratio sample-rate converter.

   Self-contained (C99, libm only). Converts between an arbitrary input
   sample rate Fs_in and output sample rate Fs_out using a windowed-sinc
   low-pass filter evaluated at polyphase offsets. Fractional delays are
   continuous, so the output keeps the original waveform timing (no pitch
   shift) and the filter is driven through its full transient on flush
   (no discontinuities at stream start/end). */

#ifndef RNNOISE_RESAMPLE_H
#define RNNOISE_RESAMPLE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Resampler Resampler;

/* Create a resampler from rate_in to rate_out.
   half_width is the sinc support in input-rate samples (taps ~ 2*half_width).
   kaiser_beta controls the window sidelobe attenuation (8.6 ~ -80 dB).
   Returns NULL on bad parameters or allocation failure. */
Resampler *resampler_create(int rate_in, int rate_out,
                            int half_width, double kaiser_beta);

void resampler_destroy(Resampler *r);

/* Feed `in_n` input samples and append converted samples to *out.
   *out_n gives capacity on entry and the produced count on return.
   Outputs accumulate starting at out[*out_n]. */
void resampler_process(Resampler *r, const float *in, size_t in_n,
                       float *out, size_t *out_n, size_t out_cap);

/* Drain the filter tail after all input was submitted. */
void resampler_flush(Resampler *r, float *out, size_t *out_n,
                     size_t out_cap);

/* Worst-case number of output samples for in_n input samples (pre-flush). */
size_t resampler_max_out(const Resampler *r, size_t in_n);

/* Total delay introduced by the converter, in output samples (fractional). */
double resampler_delay_out(const Resampler *r);

#ifdef __cplusplus
}
#endif

#endif
