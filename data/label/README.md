## Label Mapping Files

This folder contains label-to-ID mapping files used for classification tasks in the Music-EEG Cognition project. Each JSON file maps class labels (e.g., words, syllables, or genres) to their corresponding numerical indices used during model training and evaluation.

### Files

- `wordLabel.json`: Maps each Chinese word to a unique integer ID used in word-level classification.
- `syllableLabel.json`: Maps Chinese syllables to corresponding class indices for syllable-level classification.
- `genreLabel.json`: Maps music genres to numerical labels for genre classification.

> These files are essential for interpreting model outputs and preparing one-hot or categorical labels for training.

### Note

- The filenames listed previously as `*_lable_map.json` appear to have a typo: "lable" → should be "label".
- The actual filenames in this directory are correctly spelled:  
  - `wordLabel.json`  
  - `syllableLabel.json`  
  - `genreLabel.json`

Make sure to load these JSON files before training or evaluating models that depend on label mappings.
