## CNN Transfer Module

This module includes scripts used for cochleagram generation and CNN-based transfer learning for audio classification tasks.

### Files

- `cochleagram_generator.py`: Converts audio waveforms into cochleagram representations. These cochleagrams capture time-frequency auditory features inspired by the human cochlea.
- `TransferLearningCNN.py`: Defines and trains a CNN model using transfer learning. The model takes cochleagrams as input and fine-tunes classification layers for specific tasks.
- `sample_coch_'keyi'_whiteNoise_SNR5.png` is a sample cochleagram for CNN training. "keyi(可以)" in Chinese mean 'could'.

### Workflow

1. Use `cochleagram_generator.py` to preprocess audio files and generate cochleagram images.
2. Feed the generated cochleagrams into the CNN defined in `TransferLearningCNN.py` for training or evaluation.

This module forms the core of the model pipeline in the Music-EEG Cognition project.
