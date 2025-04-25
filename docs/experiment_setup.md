# Experimental Setup

## Hardware and Software Environment

All experiments were conducted on a Linux-based server equipped with:

- **Operating System**: Ubuntu 20.04 LTS  
- **CPU**: Intel Xeon Gold 6348  
- **GPU**: NVIDIA A100 40GB (CUDA 11.7)  
- **Memory**: 256 GB DDR4  
- **Framework**: TensorFlow 2.11.0 with Keras API  
- **Audio Processing Libraries**: `pycochleagram`, `librosa`, `praat-parselmouth`

## Model Architecture and Training Configuration

We used a dual-branch convolutional neural network architecture adapted from Kell et al. (2018), consisting of:

- A shared convolutional feature extractor with three layers (`conv1`, `conv2`, `conv3`)
- Two task-specific classification branches: `Word` and `Genre`
- Only the **Word Branch** was fine-tuned; the Genre Branch and shared layers were frozen

**Training hyperparameters:**

- Optimizer: Adam  
- Learning rate: `1e-4`  
- Batch size: 32  
- Epochs: 3  
- Loss: Categorical cross-entropy  
- Loss weights: `fctop_W: 1.0`, `fctop_G: 0.0`  
- Validation split: 10%

## Experimental Groups

Two independent fine-tuning setups:

- **Word Recognition**: 63-class Chinese word classification
- **Syllable Recognition**: 84-class syllable classification

## Input and Output Format

- **Input**: 256×256 cochleagram images from 16 kHz Mandarin audio clips  
  - Preprocessing: filterbank analysis, envelope compression, Lanczos interpolation
- **Output**: Softmax probability over 63 or 84 categories (task-dependent)

## Test Conditions

- **Clean**: Original speech (TTS + human)
- **Noisy**: Augmented with THCHS-30 additive noise, random SNR (5–20 dB), pitch shifting, and time-stretching

These settings were used to test model robustness and generalization across natural and degraded audio conditions.
