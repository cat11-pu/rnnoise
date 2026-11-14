# Sample-rate-aware RNNoise front end

RNNoise's model operates at a fixed **48 kHz** with 480-sample frames. The
original demo (`examples/rnnoise_demo.c`) ignores the input rate, so feeding
16 kHz or 44.1 kHz audio transposes pitch and changes duration. The tools in
this directory fix that by resampling to 48 kHz internally and back.

## Files

- `resample.h` / `resample.c` — self-contained streaming polyphase
  sample-rate converter (C99, libm only).
- `rnnoise_sr.c` — sample-rate-aware denoise command line tool.
- `rnnoise_naive.c` — reproduces the old rate-unaware behavior, kept only as
  the "before" baseline.
- `resample_pcm.c` — stand-alone PCM resampler built on `resample.c`.
- `build.sh` — builds the library and every tool without autotools.

## Resampler design

- Arbitrary rational ratio `L/M` (reduced from the two rates), so the
  44100:48000 = 147:160 case is handled exactly.
- Windowed-sinc low-pass kernel
  `h(t) = fc · sinc(pi·fc·t) · kaiser(t)`, with
  `fc = min(1, L/M) · 0.985`. The 0.985 guard band leaves room for the
  Kaiser window (beta 8.6, ~-80 dB sidelobes); the support is 48 input
  samples (half width 24).
- Continuous fractional-delay interpolation on a circular history buffer,
  zero-padded at both ends. There is no click at stream start/end because the
  filter is always driven through its full transient.
- Group delay is `half_width` input samples; the caller trims this delay so
  the output keeps the exact input duration and sample count.

Measured transparency (clean TTS, full round trip):

| conversion            | whole-signal SNR |
|-----------------------|------------------|
| 16k -> 48k -> 16k     | ~111 dB          |
| 44.1k -> 48k -> 44.1k | ~43 dB           |

## Usage

```sh
./tools/build.sh

# mono 16-bit little-endian PCM in, same rate out
./build/rnnoise_sr input_16k.raw output_16k.raw 16000
./build/rnnoise_sr input_44k.raw output_44k.raw 44100
./build/rnnoise_sr input_48k.raw output_48k.raw 48000

# any other rate is rejected instead of being misread as 48 kHz
./build/rnnoise_sr in.raw out.raw 22050   # exits with status 2

# plain resampling without denoising
./build/resample_pcm in.raw 16000 out.raw 48000
```

WAV to raw PCM and back:

```sh
ffmpeg -i in.wav -f s16le -ac 1 in.raw
ffmpeg -f s16le -ar 16000 -ac 1 -i out.raw out.wav
```

## Evaluation

`eval/run_compare.sh` regenerates all test signals and reports duration
preservation, pop/clip counts, and quality on a common 48 kHz grid.
`eval/eval_pcm.c` provides the metrics (segmental and whole-signal SNR,
relative MOS, isolated-spike pop detector). On clean speech every supported
rate scores the same MOS as the native 48 kHz path (gap 0.0) with zero pops.
