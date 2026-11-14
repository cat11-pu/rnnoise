#!/usr/bin/env python3
"""Generate synthetic noisy-speech PCM corpus for VAD evaluation.

No third-party dependencies. Produces 48 kHz mono signed-16 PCM plus a
per-frame (10 ms = 480 samples) label file where 1 marks speech frames.

Speech is modelled as a pitch-excited source passed through time-varying
formant resonators; interference is shaped (pink/brown) noise. The generated
material is intended for pipeline and metric validation, not as a substitute
for recorded corpora.
"""

import math
import os
import random
import struct
import sys

SAMPLE_RATE = 48000
FRAME_SIZE = 480


class PinkNoise:
    """Voss-McCartney style pink noise with an extra low-pass tilt."""

    def __init__(self, rng):
        self.rng = rng
        self.rows = [rng.uniform(-1.0, 1.0) for _ in range(16)]
        self.counter = 0
        self.tilt_state = 0.0

    def sample(self):
        self.counter += 1
        white = self.rng.uniform(-1.0, 1.0)
        for bit in range(16):
            if self.counter & (1 << bit):
                self.rows[bit] = white
                break
        pink = sum(self.rows) * 0.06
        # One-pole low-pass adds a brown component for rumble realism.
        self.tilt_state += 0.02 * (pink - self.tilt_state)
        return pink + 0.5 * self.tilt_state


class FormantFilter:
    """Cascade of biquad resonators approximating vocal tract formants."""

    def __init__(self, freqs, q):
        self.sections = []
        for f0 in freqs:
            w0 = 2.0 * math.pi * f0 / SAMPLE_RATE
            alpha = math.sin(w0) / (2.0 * q)
            b0 = 1.0 + alpha
            b = [alpha / b0, 0.0, -alpha / b0]
            a = [1.0, -2.0 * math.cos(w0) / b0, (1.0 - alpha) / b0]
            self.sections.append((b, a, [0.0, 0.0], [0.0, 0.0]))

    def process(self, x):
        for b, a, xmem, ymem in self.sections:
            y = b[0] * x + xmem[0]
            xmem[0] = b[1] * x - a[1] * y + xmem[1]
            xmem[1] = b[2] * x - a[2] * y
            x = y
        return x


def synth_speech_segment(rng, n_frames, base_amp):
    """Return one float sample per frame slot with voiced/unvoiced texture."""
    n = n_frames * FRAME_SIZE
    out = [0.0] * n
    f0 = rng.uniform(100.0, 180.0)
    formants = [rng.uniform(700, 900), rng.uniform(1100, 1400),
                rng.uniform(2400, 2900)]
    vf = FormantFilter(formants, 6.0)
    phase = 0.0
    amp_env = []
    # Slow amplitude and pitch contours so segments are not stationary tones.
    contour_period = rng.uniform(0.18, 0.4)
    for i in range(n):
        t = i / SAMPLE_RATE
        contour = 0.65 + 0.35 * math.sin(2 * math.pi * t / contour_period)
        amp_env.append(base_amp * contour)
        jitter = 1.0 + 0.01 * math.sin(2 * math.pi * 7.3 * t)
        phase += 2 * math.pi * f0 * jitter / SAMPLE_RATE
        phase %= 2 * math.pi
        # Glottal-ish source: harmonics with a spectral tilt.
        source = (math.sin(phase) + 0.45 * math.sin(2 * phase)
                  + 0.2 * math.sin(3 * phase))
        if rng.random() < 0.02:
            source += rng.uniform(-0.8, 0.8)
        out[i] = vf.process(source) * amp_env[i]
    # Gentle onset/offset ramps avoid step transients at boundaries.
    ramp = int(0.02 * SAMPLE_RATE)
    for i in range(ramp):
        scale = i / ramp
        out[i] *= scale
        out[n - 1 - i] *= scale
    return out


def generate(path_pcm, path_labels, duration_s, noise_amp, speech_amp, seed):
    rng = random.Random(seed)
    noise = PinkNoise(rng)
    total_frames = int(duration_s * SAMPLE_RATE / FRAME_SIZE)
    labels = [0] * total_frames
    samples = [0.0] * (total_frames * FRAME_SIZE)

    # Continuous background noise over the whole clip.
    for i in range(len(samples)):
        samples[i] = noise_amp * noise.sample()

    # Speech bursts with random start and length, never touching the edges.
    frame = 2
    while speech_amp > 0.0 and frame < total_frames - 4:
        if rng.random() < 0.55:
            length = int(rng.uniform(0.25, 0.9) * SAMPLE_RATE / FRAME_SIZE)
            length = min(length, total_frames - frame - 2)
            voiced = synth_speech_segment(rng, length, speech_amp)
            start_sample = frame * FRAME_SIZE
            for j, value in enumerate(voiced):
                samples[start_sample + j] += value
            for k in range(length):
                labels[frame + k] = 1
            frame += length + int(rng.uniform(0.2, 0.6) * SAMPLE_RATE / FRAME_SIZE)
        else:
            frame += int(rng.uniform(0.2, 0.5) * SAMPLE_RATE / FRAME_SIZE)

    peak = max(1e-9, max(abs(x) for x in samples))
    scale = 0.9 / peak
    with open(path_pcm, "wb") as f:
        f.write(b"".join(struct.pack("<h", int(max(-1.0, min(1.0, x * scale)) * 32767))
                         for x in samples))
    with open(path_labels, "w") as f:
        f.write("\n".join(str(label) for label in labels) + "\n")
    speech_frames = sum(labels)
    print(f"{os.path.basename(path_pcm)}: frames={total_frames} "
          f"speech={speech_frames} silence={total_frames - speech_frames} "
          f"noise_amp={noise_amp} speech_amp={speech_amp} seed={seed}")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    # Three mixed SNR-ish conditions plus a pure-noise clip (all-silence
    # ground truth) to stress false alarms.
    cases = [
        ("vad_mix_low.pcm",  10.0, 0.16, 0.22, 101),
        ("vad_mix_mid.pcm",  20.0, 0.10, 0.24, 202),
        ("vad_mix_high.pcm", 30.0, 0.05, 0.26, 303),
        ("vad_noise.pcm",    20.0, 0.14, 0.00, 404),
    ]
    label_names = []
    for name, duration, noise_amp, speech_amp, seed in cases:
        pcm = os.path.join(out_dir, name)
        labels = pcm.replace(".pcm", ".labels")
        generate(pcm, labels, duration, noise_amp, speech_amp, seed)
        label_names.append(labels)


if __name__ == "__main__":
    main()
