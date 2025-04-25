## Dataset Directory

This folder contains data used in the **Music-EEG Cognition** project, including stimulus audio files and label mappings for classification tasks.

---

### `music_audio/`  
Contains **test music stimuli** in MP3 format and their corresponding **TextGrid** annotations. These are used during the **testing phase** or in **EEG experiments**.

Each stimulus consists of:
- `musicX.mp3`: The audio file (speed-adjusted variants such as 1.0x, 2.0x, etc.)
- `musicX.TextGrid`: Time-aligned annotations for phoneme or word-level content (Praat format)

> **Note**: MP3 files must be converted to **16kHz .wav format** before use in preprocessing or CNN-based models.

---

### `label/`  
Includes JSON files mapping human-readable labels (words, syllables, genres) to numeric class indices used for training and evaluating the model.

- `wordLabel.json`: Word-to-index mapping for word-level classification  
- `syllableLabel.json`: Syllable-to-index mapping for syllable-level classification  
- `genreLabel.json`: Genre-to-index mapping for music genre classification

These files are essential for decoding model predictions and constructing categorical training labels.

---

### Usage Flow Overview

1. **Audio Processing**  
   Convert `.mp3` files in `music_audio/` to `.wav` format (16kHz) for input.

2. **Label Alignment**  
   Use `.TextGrid` files for aligning and segmenting input audio with linguistic annotations.

3. **Model Training/Evaluation**  
   Load appropriate label maps from `label/` when training or interpreting outputs from word/syllable/genre CNN classifiers.

---

For more details on how these data files are used in training and evaluation, please refer to the corresponding modules under `/modules/`.
