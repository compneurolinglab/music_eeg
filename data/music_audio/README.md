## Music Audio Stimuli (Test Set)

This folder contains **test music stimuli** used for model evaluation, feature extraction and EEG experiments in the Music-EEG Cognition project.

### Contents

Each test stimulus includes:

- An **MP3 audio file** (`musicX.mp3`) containing the music stimulus.
- A corresponding **TextGrid file** (`musicX.TextGrid`) with phoneme-, syllable- and word-level time-alignment annotations, compatible with **Praat**.

Example:
- `music1.mp3` ⟶ `music1.TextGrid`
- `music2.mp3` ⟶ `music2.TextGrid`
- ...

> **NOTE**: Audio files provided are in MP3 format. For further processing (e.g., cochleagram generation or feature extraction), they need to be converted to **16kHz .wav format**.

### Stimulus Variants

The test stimuli is played at different speeds (e.g., 1.0x, 2.0x, 3.0x, 4.0x) in behavioral or EEG settings. Please ensure consistent alignment between speed-adjusted audio and its corresponding annotation file.

---

These files are designed for use during testing, model inference and feature(activation value) extraction only. For training stimuli and label mappings, please refer to the `/data/label/` and other relevant folders.
