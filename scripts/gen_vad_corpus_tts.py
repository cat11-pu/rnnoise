#!/usr/bin/env python3
"""Build a labelled noisy-speech corpus from flite TTS and shaped noise.

Requires ffmpeg with the libflite filter. Outputs 48 kHz mono s16 PCM and a
per-10-ms-frame label file (1 = speech frame). Speech clip positions are
known by construction, so labels need no manual annotation.
"""

import math
import os
import random
import struct
import subprocess
import sys
import wave

SAMPLE_RATE = 48000
FRAME_SIZE = 480

UTTERANCES = [
    "the quick brown fox jumps over the lazy dog",
    "voice activity detection separates speech from background noise",
    "one two three four five six seven eight nine ten",
    "meeting notes are transcribed after the call ends",
    "please lower the music while the speaker is talking",
    "the train arrives at the station every fifteen minutes",
]


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
        assert w.getsampwidth() == 2
        raw = w.readframes(w.getnframes())
    return [float(x) / 32768.0 for x in struct.unpack("<%dh" % (len(raw) // 2), raw)]


def write_wav(path, samples):
  data = struct.pack("<%dh" % len(samples),
                     *[int(max(-1.0, min(1.0, x)) * 32767) for x in samples])
  with open(path, "wb") as f:
      f.write(data)


class PinkNoise:
    def __init__(self, rng):
        self.rng = rng
        self.rows = [rng.uniform(-1, 1) for _ in range(16)]
        self.counter = 0
        self.tilt = 0.0

    def sample(self):
        self.counter += 1
        white = self.rng.uniform(-1, 1)
        for bit in range(16):
            if self.counter & (1 << bit):
                self.rows[bit] = white
                break
        pink = sum(self.rows) * 0.06
        self.tilt += 0.02 * (pink - self.tilt)
        return pink + 0.5 * self.tilt


def synthesize(tmp_dir, text, index):
    path = os.path.join(tmp_dir, "tts_%d.wav" % index)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "flite=text=%s" % text,
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", path,
    ]
    subprocess.run(cmd, check=True)
    return read_wav(path)


def rms(samples):
    return math.sqrt(sum(x * x for x in samples) / max(1, len(samples)))


def trim_voiced(samples):
    """Crop leading/trailing near-silence, keeping the full voiced body."""
    window = FRAME_SIZE
    frame_count = len(samples) // window
    energies = [rms(samples[f * window:(f + 1) * window])
                for f in range(frame_count)]
    peak = max(energies) if energies else 0.0
    floor = max(0.2 * peak, 1e-4)
    first = 0
    while first < frame_count and energies[first] < floor:
        first += 1
    last = frame_count
    while last > first and energies[last - 1] < floor:
        last -= 1
    if first >= last:
        return samples
    return samples[first * window:last * window]


def build(out_dir, name, duration_s, snr_db, seed):
    rng = random.Random(seed)
    total = duration_s * SAMPLE_RATE
    mix = [0.0] * total
    labels = [0] * (total // FRAME_SIZE)

    # Place utterances at frame-aligned offsets with random silence gaps.
    cursor = SAMPLE_RATE // 2
    clip_index = 0
    speech_rms_values = []
    while cursor < total - SAMPLE_RATE:
        clip = synthesize(out_dir, UTTERANCES[clip_index % len(UTTERANCES)],
                          clip_index)
        clip_index += 1
        clip = trim_voiced(clip)
        level = rng.uniform(0.8, 1.0)
        clip = [x * level for x in clip]
        speech_rms_values.append(rms(clip))
        end = min(cursor + len(clip), total)
        for i in range(cursor, end):
            mix[i] += clip[i - cursor]
        # The clip is silence-cropped; all its frames are one speech unit.
        frame_count = (end - cursor) // FRAME_SIZE
        base_frame = cursor // FRAME_SIZE
        for frame in range(frame_count):
            labels[base_frame + frame] = 1
        # RNNoise keeps reporting speech for roughly 80-120 ms while the
        # utterance tail decays; extend the label to cover that latency.
        for frame in range(frame_count, min(frame_count + 12,
                                            len(labels) - base_frame)):
            labels[base_frame + frame] = 1
        cursor = end + int(rng.uniform(0.7, 1.3) * SAMPLE_RATE)

    # Scale the continuous noise to the requested SNR relative to speech RMS.
    speech_rms_value = sum(speech_rms_values) / len(speech_rms_values)
    noise_only = PinkNoise(random.Random(seed + 7))
    noise_track = [noise_only.sample() for _ in range(total)]
    noise_rms_value = rms(noise_track)
    noise_scale = speech_rms_value / (noise_rms_value * (10.0 ** (snr_db / 20.0)))
    for i in range(total):
        mix[i] += noise_track[i] * noise_scale

    peak = max(1e-9, max(abs(x) for x in mix))
    mix = [x * (0.9 / peak) for x in mix]
    pcm_path = os.path.join(out_dir, name + ".pcm")
    label_path = os.path.join(out_dir, name + ".labels")
    write_wav(pcm_path, mix)
    with open(label_path, "w") as f:
        f.write("\n".join(str(x) for x in labels) + "\n")
    speech_frames = sum(labels)
    print("%s: frames=%d speech=%d silence=%d snr=%ddB seed=%d"
           % (name, len(labels), speech_frames, len(labels) - speech_frames,
              snr_db, seed))


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    build(out_dir, "tts_snr_0", 30, 0, 11)
    build(out_dir, "tts_snr_5", 30, 5, 22)
    build(out_dir, "tts_snr_10", 30, 10, 33)
    build(out_dir, "tts_snr_15", 30, 15, 44)


if __name__ == "__main__":
    main()
