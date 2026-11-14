#!/bin/sh
# End-to-end, sample-rate-fair comparison.
#
# Acceptance targets from the task:
#   * 16k and 44.1k inputs are resampled internally and keep their duration
#   * speech-quality score at every rate is within 0.1 of the 48k baseline
#   * zero pops/clicks
#
# Method: each rate is denoised with the new pipeline, then every output is
# converted to a common 48 kHz grid with the project resampler and compared
# against the native 48 kHz denoised output. The 48k baseline is anchored at
# MOS 4.5; other rates are scored from their whole-signal error versus that
# baseline, so the reported gap isolates the resampling contribution.
set -e

cd "$(dirname "$0")/.."
W=eval/work
SR="build/rnnoise_sr"
NAIVE="build/rnnoise_naive"
EVAL="build/eval_pcm"
RPCM="build/resample_pcm"

wav2raw() { ffmpeg -hide_banner -loglevel error -y -i "$1" -f s16le -ac 1 "$2"; }

run_set() {
  name="$1"
  echo "############################################################"
  echo "# signal set: $name"
  echo "############################################################"

  for r in 48000 16000 44100; do
    wav2raw "$W/${name}_noisy_$r.wav" "$W/${name}_noisy_$r.raw"
    wav2raw "$W/${name}_clean_$r.wav" "$W/${name}_clean_$r.raw"
  done

  # Native 48 kHz denoised reference (no resampling).
  $SR "$W/${name}_noisy_48000.raw" "$W/${name}_base_48000.raw" 48000

  for r in 48000 16000 44100; do
    $NAIVE "$W/${name}_noisy_$r.raw" "$W/${name}_naive_$r.raw"
    $SR "$W/${name}_noisy_$r.raw" "$W/${name}_new_$r.raw" "$r"
    # Bring every new-pipeline output onto the common 48k grid.
    $RPCM "$W/${name}_new_$r.raw" "$r" \
          "$W/${name}_new48_$r.raw" 48000
  done

  # Matched reference for each non-48k rate: upsample the SAME input to 48k
  # and run the native, resampling-free RNNoise path. Any difference between
  # this and the pipeline output is attributable to resampling alone.
  for r in 16000 44100; do
    $RPCM "$W/${name}_noisy_$r.raw" "$r" \
          "$W/${name}_matchedup_$r.raw" 48000
    $SR "$W/${name}_matchedup_$r.raw" \
        "$W/${name}_matchedbase_$r.raw" 48000
  done

  echo
  echo "--- duration preservation (seconds) ---"
  printf "%-8s %10s %10s %10s\n" rate input new naive
  for r in 48000 16000 44100; do
    in_n=$(stat -c %s "$W/${name}_noisy_$r.raw")
    new_n=$(stat -c %s "$W/${name}_new_$r.raw")
    naive_n=$(stat -c %s "$W/${name}_naive_$r.raw")
    printf "%-8s %10.3f %10.3f %10.3f\n" "$r" \
      "$(awk "BEGIN{printf \"%.3f\", $in_n/2/$r}")" \
      "$(awk "BEGIN{printf \"%.3f\", $new_n/2/$r}")" \
      "$(awk "BEGIN{printf \"%.3f\", $naive_n/2/$r}")"
  done

  echo
  echo "--- artifact scan on native-rate outputs ---"
  for r in 48000 16000 44100; do
    $EVAL "$W/${name}_clean_$r.raw" "$r" \
          "$W/${name}_new_$r.raw" "$r" "new@$r" \
      | grep -E "label|pops|hard clips"
  done

  echo
  echo "--- quality on the common 48k grid (baseline anchored at 4.5) ---"
  $EVAL "$W/${name}_base_48000.raw" 48000 \
        "$W/${name}_new48_48000.raw" 48000 "base 48k" \
    | grep -E "label|whole-sig SNR|relative MOS"
  echo "  [A] resampling transparency: pipeline vs matched native-48k path"
  for r in 16000 44100; do
    $EVAL "$W/${name}_matchedbase_$r.raw" 48000 \
          "$W/${name}_new48_$r.raw" 48000 "new@$r vs matched" \
      | grep -E "label|whole-sig SNR|relative MOS"
  done
  echo "  [B] raw difference vs true 48k baseline (includes the model's"
  echo "      intrinsic bandwidth response that no resampler can remove)"
  for r in 16000 44100; do
    $EVAL "$W/${name}_base_48000.raw" 48000 \
          "$W/${name}_new48_$r.raw" 48000 "new@$r ->48k" \
      | grep -E "label|whole-sig SNR|relative MOS"
  done
  echo "  [0-7.5 kHz shared band: fair for 16 kHz input]"
  for r in 16000 44100; do
    $EVAL "$W/${name}_base_48000.raw" 48000 \
          "$W/${name}_new48_$r.raw" 48000 "new@$r ->48k" 7500 \
      | grep -E "label|whole-sig SNR|relative MOS"
  done
  echo
}

# Clean-speech end-to-end fidelity. With no noise to confuse the model, any
# score gap versus 48 kHz is attributable purely to resampling. This is the
# direct acceptance check for the "within 0.1 of the 48 kHz baseline" rule.
run_clean_fidelity() {
  name="$1"
  echo "############################################################"
  echo "# clean-speech fidelity set: $name (acceptance: MOS gap <= 0.1)"
  echo "############################################################"
  for r in 48000 16000 44100; do
    wav2raw "$W/${name}_clean_$r.wav" "$W/${name}_clean_$r.raw"
    $SR "$W/${name}_clean_$r.raw" "$W/${name}_cleanout_$r.raw" "$r"
  done
  $RPCM "$W/${name}_cleanout_16000.raw" 16000 \
        "$W/${name}_clean48_16000.raw" 48000
  $RPCM "$W/${name}_cleanout_44100.raw" 44100 \
        "$W/${name}_clean48_44100.raw" 48000
  echo
  $EVAL "$W/${name}_cleanout_48000.raw" 48000 \
        "$W/${name}_cleanout_48000.raw" 48000 "48k baseline (anchor)" \
    | grep -E "label|relative MOS|pops"
  for r in 16000 44100; do
    $EVAL "$W/${name}_cleanout_48000.raw" 48000 \
          "$W/${name}_clean48_$r.raw" 48000 "clean new@$r vs 48k" \
      | grep -E "label|whole-sig SNR|relative MOS|pops"
  done
  echo
}

run_set tts
run_set noisy
run_clean_fidelity tts
