# Experimental Documentation

## Experimental Setup

### Hardware and Software Environment
All experiments were conducted on a Linux-based server equipped with:

- **Operating System**: Ubuntu 20.04 LTS  
- **CPU**: Intel Xeon Gold 6348  
- **GPU**: NVIDIA A100 40GB (CUDA 11.7)  
- **Memory**: 256 GB DDR4  
- **Framework**: TensorFlow 2.11.0 with Keras API  
- **Audio Processing Libraries**: pycochleagram, librosa, praat-parselmouth  

### Model Architecture and Training Configuration
We used a dual-branch convolutional neural network architecture adapted from Kell et al. (2018), consisting of:

- A shared convolutional feature extractor with three layers (conv1, conv2, conv3)
- Two task-specific classification branches: **Word** and **Genre**
- Only the Word Branch was fine-tuned; the Genre Branch and shared layers were frozen

**Training hyperparameters**:

- Optimizer: Adam  
- Learning rate: 1e-4  
- Batch size: 32  
- Epochs: 3  
- Loss: Categorical cross-entropy  
- Loss weights: fctop_W: 1.0, fctop_G: 0.0  
- Validation split: 10%  

### Experimental Groups
Two independent fine-tuning setups:

- **Word Recognition**: 63-class Chinese word classification  
- **Syllable Recognition**: 84-class syllable classification  

### Input and Output Format
- **Input**: 256×256 cochleagram images from 16 kHz Mandarin audio clips  
- **Preprocessing**: filterbank analysis, envelope compression, Lanczos interpolation  
- **Output**: Softmax probability over 63 or 84 categories (task-dependent)  

### Test Conditions
- **Clean**: Original speech (TTS + human)  
- **Noisy**: Augmented with THCHS-30 additive noise, random SNR (5–20 dB), pitch shifting, and time-stretching  

## Data Collection and Annotation Procedures

### Audio Sources
Two types of speech data were used for training and evaluation:

- **Synthetic Speech**:  
  Generated using Edge TTS (Microsoft Azure) and Google TTS engines. Both Mandarin Chinese female and male voices were used for diversity.

- **Human Recordings**:  
  A set of Chinese volunteers read and recorded selected song lyrics in a quiet environment. Recordings were performed using high-quality USB microphones at a 16 kHz sampling rate.

### Text Materials
The linguistic content used for all speech stimuli was derived from a single full-length Mandarin Chinese song. The lyrics of this song were used consistently across all conditions and speakers.

Two classification tasks were constructed by annotating the lyrics at different linguistic levels:

- **Word Task**:  
  Based on word-level annotations using Praat. Used to train a dual-pathway CNN for word classification + genre classification.

- **Syllable Task**:  
  Based on syllable-level annotations using Praat. Used to train a separate dual-pathway CNN for syllable classification + genre classification.

This setup ensured controlled comparison of word- and syllable-level decoding while preserving consistent acoustic and lexical content.

### Annotation Procedure
Each audio clip was segmented and labeled with the following information:

- **Syllable Boundaries**:  
  Time-aligned using forced alignment scripts and manually corrected using Praat.

- **Word Onsets**:  
  Estimated from the original text structure and verified against the aligned audio in Praat.

- **Class Labels**:
  - For Word Task: 63-class Chinese word labels
  - For Syllable Task: 84-class Chinese syllable labels  
  Labels were encoded into one-hot vectors using a `label_map.json` file.

### Tools and Automation

- **Praat + Parselmouth**:  
  Used for manual alignment correction and boundary visualization.

- **Edge TTS Scripting**:  
  Automatically generated WAV files from textual prompts with boundary metadata.

- **NumPy Pipelines**:  
  Converted aligned audio into cochleagrams and stored them with synchronized metadata.

### Annotation Quality Control
- **Annotators**:  
  3 bilingual Mandarin speakers with linguistic training were involved in reviewing boundaries.

- **Validation Process**:  
  Each segment was reviewed by two annotators. In case of mismatch, a third annotator resolved the conflict. Final boundary error tolerance was controlled within ±20 ms.

### Label Extraction and Encoding
All training data were automatically labeled based on the filenames of the preprocessed cochleagram `.npy` files.

Each `.npy` file represents one stimulus segment derived from the same Chinese song, and filenames follow a structured convention such as:

```
我_1.npy, 我_2.npy, 可以.npy, 沙滩.npy
```

To generate labels for classification:

- The unique Chinese word (or syllable) was extracted from the filename prefix.
- A sorted list of all unique labels was constructed.
- Each label was assigned a unique integer ID.
- The resulting dictionary (e.g., `{ "我": 0, "可以": 1, "沙滩": 2 }`) was saved to `label_map.json`.

**Python script example**:
```python
import os
import json

npy_folder = "autodl-tmp/TrainDataSet/cochleagrams_npy"
npy_files = [f.replace('.npy', '') for f in os.listdir(npy_folder) if f.endswith('.npy')]
label_dict = {word: idx for idx, word in enumerate(sorted(npy_files))}

with open("autodl-tmp/TrainDataSet/labels/label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_dict, f, ensure_ascii=False, indent=4)

print("finished")
```

This label map was used for one-hot encoding in both the word and syllable classification tasks.


# Music-EEG-Cognition

This project explores the neural channel capacity of the human brain in processing music and speech using EEG data and deep learning models.

Inspired by previous studies in auditory cognition and information theory, we investigate how the brain allocates resources when decoding musical genres versus linguistic content under varying playback conditions. The framework integrates EEG-based experiments, linguistic feature analysis, and a dual-pathway CNN model to simulate auditory perception and establish parallels between biological and computational processing limits.

---

## Project Structure

```bash
Music-EEG-Cognition/
├── data/                # Dataset and labels
│   ├── music_audio/     # Test audio stimuli (.mp3) and TextGrid annotations
│   └── label/           # JSON label mappings (word, syllable, genre)
│
├── modules/             # Core processing modules
│   └── Demo/            # Jupyter notebooks for demo: TTS, feature extraction, prediction
│
├── docs/                # Documentation and architecture diagrams
└── README.md            # Project overview and instructions
```

---

## Key Features

- **Dual-pathway CNN model** for music and linguistic auditory decoding  
- **EEG + audio integration** to study neural resource allocation  
- **Cochleagram-based preprocessing** mimicking the human cochlea  
- **Transfer learning** with fine-tuned convolutional networks  
- **Support for speed-modulated stimuli** (1x, 2x, 3x, 4x)  

---

## Getting Started

1. Convert audio `.mp3` files in `data/music_audio/` to 16kHz `.wav`  
2. Generate cochleagrams using `cochleagram_generator.py`  
3. Train or evaluate CNN models with `TransferLearningCNN.py`  
4. Use label mappings in `data/label/` for classification outputs  
5. Explore feature pipelines via notebooks in `modules/Demo/`  

---

## Documentation

Refer to the `docs/` folder for:
- CNN architecture diagrams  
- Experimental setup  
- Data collection and annotation procedures  

---

## Dataset Directory

This folder contains data used in the Music-EEG Cognition project, including stimulus audio files and label mappings for classification tasks.

### music_audio/
Contains test music stimuli in MP3 format and their corresponding TextGrid annotations. These are used during the testing phase or in EEG experiments.

Each stimulus consists of:

- `musicX.mp3`: The audio file (speed-adjusted variants such as 1.0x, 2.0x, etc.)  
- `musicX.TextGrid`: Time-aligned annotations for phoneme or word-level content (Praat format)  

> Note: MP3 files must be converted to 16kHz .wav format before use in preprocessing or CNN-based models.

### label/
Includes JSON files mapping human-readable labels (words, syllables, genres) to numeric class indices used for training and evaluating the model.

- `wordLabel.json`: Word-to-index mapping for word-level classification  
- `syllableLabel.json`: Syllable-to-index mapping for syllable-level classification  
- `genreLabel.json`: Genre-to-index mapping for music genre classification  

These files are essential for decoding model predictions and constructing categorical training labels.

---

## Usage Flow Overview

### Audio Processing
Convert `.mp3` files in `music_audio/` to `.wav` format (16kHz) for input.

### Label Alignment
Use `.TextGrid` files for aligning and segmenting input audio with linguistic annotations.

### Model Training/Evaluation
Load appropriate label maps from `label/` when training or interpreting outputs from word/syllable/genre CNN classifiers.

> For more details on how these data files are used in training and evaluation, please refer to the corresponding modules under `/modules/`.

---

## Modules

This directory contains modular components used in the Music-EEG Cognition project. Each submodule performs a specific stage in the overall processing pipeline, including data preprocessing, feature extraction, transfer learning, and demo inference.

### Submodules

#### Demo/
Includes interactive notebooks demonstrating how to process audio data, extract features, synthesize speech, and perform CNN-based predictions.

- `feature_extraction.ipynb`: Extracts acoustic features from audio stimuli.  
- `merge_features.ipynb`: Merges feature sets for model input.  
- `textToSpeech.ipynb`: Converts text into synthetic speech using TTS.  
- `demo_cnn_prediction.ipynb`: Shows prediction results from the trained CNN model.  

> Note: Audio files should be converted to 16kHz `.wav` format for compatibility with feature extraction and model input.

#### cnn_transfer/
Contains core scripts for training the CNN using transfer learning and preprocessing audio into cochleagram representations.

- `cochleagram_generator.py`: Generates cochleagram images from audio signals.  
- `TransferLearningCNN.py`: Defines and fine-tunes the CNN model using transfer learning techniques.  

> The cochleagrams generated by `cochleagram_generator.py` should be converted into `.npy` format before used as input to the CNN defined in `TransferLearningCNN.py`.

Each module is designed to be modular and reusable across different stages of the pipeline. For further details, refer to the README files in each subfolder.



---

## Transfer Learning CNN Code

```python
import os
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.utils import to_categorical

# Define the original CNN architecture (Kell et al., 2018)
def build_kell2018_cnn(input_shape=(256, 256, 1), num_classes_word=589, num_classes_genre=43):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Shared layers
    x = tf.keras.layers.Conv2D(96, (9,9), strides=3, activation='relu', padding='same', name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same', name="pool1")(x)

    x = tf.keras.layers.Conv2D(256, (5,5), strides=2, activation='relu', padding='same', name="conv2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same', name="pool2")(x)

    x = tf.keras.layers.Conv2D(512, (3,3), strides=1, activation='relu', padding='same', name="conv3")(x)

    # Word Recognition Branch (to be fine-tuned)
    x_w = tf.keras.layers.Conv2D(1024, (3,3), strides=1, activation='relu', padding='same', name="conv4_W")(x)
    x_w = tf.keras.layers.Conv2D(512, (3,3), strides=1, activation='relu', padding='same', name="conv5_W")(x_w)
    x_w = tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=2, padding='same', name="pool5_W")(x_w)

    x_w = tf.keras.layers.Flatten()(x_w)
    x_w = tf.keras.layers.Dense(1024, activation='relu', name="fc6_W")(x_w)
    output_word = tf.keras.layers.Dense(num_classes_word, activation='softmax', name="fctop_W")(x_w)

    # Genre Recognition Branch (kept unchanged)
    x_g = tf.keras.layers.Conv2D(1024, (3,3), strides=1, activation='relu', padding='same', name="conv4_G")(x)
    x_g = tf.keras.layers.Conv2D(512, (3,3), strides=1, activation='relu', padding='same', name="conv5_G")(x_g)
    x_g = tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=2, padding='same', name="pool5_G")(x_g)

    x_g = tf.keras.layers.Flatten()(x_g)
    x_g = tf.keras.layers.Dense(1024, activation='relu', name="fc6_G")(x_g)
    output_genre = tf.keras.layers.Dense(num_classes_genre, activation='softmax', name="fctop_G")(x_g)

    model = tf.keras.models.Model(inputs=inputs, outputs=[output_word, output_genre])
    return model

# (Omitted for brevity) ... Model loading, training, and saving processes follow
```



---

## Cochleagram Generation Code

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal
import multiprocessing
from pycochleagram import cochleagram as cgram
from PIL import Image

input_folder = "cnnMusicSeg/music1_Seg"
output_folder = "cnnMusicSeg/pngMusic1_Seg"

os.makedirs(output_folder, exist_ok=True)

def resample(example, new_size):
    im = Image.fromarray(example)
    resized_image = im.resize(new_size, resample=Image.Resampling.LANCZOS)
    return np.array(resized_image)

def plot_cochleagram(cochleagram, output_path):
    plt.figure(figsize=(6, 3))
    plt.imshow(cochleagram, origin='lower', cmap=plt.cm.Blues, aspect='auto')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_cochleagram(wav_f, sr):
    n, low_lim, hi_lim = 50, 20, 8000
    sample_factor, pad_factor = 4, 2
    downsample = 250 if sr % 250 == 0 else 200
    nonlinearity, fft_mode, ret_mode = 'power', 'auto', 'envs'

    num_samples = int(np.ceil(len(wav_f) / (sr / downsample)) * (sr / downsample))
    wav_f = librosa.util.fix_length(wav_f, size=num_samples)

    if sr % downsample != 0:
        wav_f = scipy.signal.resample_poly(wav_f, downsample, sr)
        sr = downsample

    c_gram = cgram.cochleagram(wav_f, sr, n, low_lim, hi_lim, 
                               sample_factor, pad_factor, downsample,
                               nonlinearity, fft_mode, ret_mode, strict=True)

    c_gram_rescaled = 255 * (1 - ((np.max(c_gram) - c_gram) / np.ptp(c_gram)))

    c_gram_reshape = resample(c_gram_rescaled, (256, 256))

    return c_gram_reshape

def process_single_wav(input_wav_path, output_image_path):
    try:
        wav_f, sr = librosa.load(input_wav_path, sr=16000)
        cochleagram_img = generate_cochleagram(wav_f, sr)
        plot_cochleagram(cochleagram_img, output_image_path)
        print(f"finished: {os.path.basename(input_wav_path)} -> {output_image_path}")
    except Exception as e:
        print(f"failed: {os.path.basename(input_wav_path)} - {e}")

def process_wav_folder_parallel(input_folder, output_folder, num_workers=1):
    wav_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]

    print(f" {len(wav_files)} WAV files, processing with {num_workers} cores...")

    file_pairs = [(os.path.join(input_folder, f), os.path.join(output_folder, f.replace(".wav", ".png")))
                  for f in wav_files]

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(process_single_wav, file_pairs)

    print("All done!")

if __name__ == "__main__":
    process_wav_folder_parallel(input_folder, output_folder, num_workers=5)
```

## Acknowledgements

This project utilizes code from [mcdermottLab/kelletal2018](https://github.com/mcdermottLab/kelletal2018), developed by Kell et al. for their research published in *Neuron* (2018). We thank the authors for making their code available.

---

## Contact

For questions or contributions, feel free to open an issue or pull request.

---

**License**: [MIT](LICENSE)  
