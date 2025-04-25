## Music Audio Stimuli (Test Set)

This folder contains **test music stimuli** used for model evaluation and EEG experiments in the Music-EEG Cognition project.

### Contents

Each test stimulus includes:

- An **MP3 audio file** (`musicX.mp3`) containing the music stimulus.
- A corresponding **TextGrid file** (`musicX.TextGrid`) with phoneme or word-level time-alignment annotations, compatible with **Praat** software.

Example:
- `music1.mp3` ⟶ `music1.TextGrid`
- `music2.mp3` ⟶ `music2.TextGrid`
- ...

> **NOTE**: The audio files are in MP3 format. For further processing (e.g., cochleagram generation or feature extraction), they need to be converted to **16kHz WAV format**.

### Stimulus Variants

The test stimuli may be played at different speeds (e.g., 1.0x, 2.0x, 3.0x, 4.0x) in behavioral or EEG settings. Please ensure consistent alignment between speed-adjusted audio and its corresponding label file.

---

These files are designed for use during testing or model inference only. For training stimuli and label mappings, refer to the `/data/label/` and other relevant folders.
