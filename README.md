# Music_EEG

This project explores the neural channel capacity of the human brain in processing music and speech using EEG data and deep learning models.

Inspired by previous studies in auditory cognition and information theory, we investigate how the brain allocates resources when decoding musical genres versus linguistic content under varying playback conditions. The framework integrates EEG-based experiments, linguistic feature analysis, and a dual-pathway CNN model to simulate auditory perception and establish parallels between biological and computational processing limits.

---

## Project Structure

```bash
Music-EEG-Cognition/
├── data/                # Dataset and labels (syllable)
│   ├── music_audio/     # Test audio stimuli (.mp3) and corresponding TextGrid annotations
│   └── label/           # JSON label mappings (syllable, genre)
│
├── modules/             # Core processing modules
│   └── Demo/            # Jupyter notebooks for demo: TTS for dataset generation, prediction, feature extraction
│
├── docs/                # Documentation and architecture diagrams
|
└── README.md            # Project overview and instructions
```

---

## Overview

- AI-synthesized 2-minute Mandarin song: "智联家园"
- Playback at 4 speeds: 1×, 2×, 3×, 4×
- High-density 256-channel EEG from 36 native Mandarin speakers
- 263 syllables per speed, segmented using Whisper Large-V3
- Syllable-level CNN features aligned to EEG via multivariate temporal response function (mTRF)

---

## Key Features

| Module | Description |
|--------|-------------|
| **Stimulus Processing** | Whisper V3 used to segment 263 syllables at each speed |
| **Audio Features** | Cochleagram features extracted and fed to fine-tuned CNN |
| **EEG Recording** | 256-channel EEG while listening at 4 playback speeds |
| **CNN Layers** | Activations from `conv1` to `conv5_G` and `conv5_S` |
| **mTRF Analysis** | Align EEG with CNN-derived features (PCA-reduced) |
| **Statistical Tests** | Cluster-based permutation (10k reps) using MNE & Eelbrain |
| **Performance Evaluation** | Top-5 accuracy on syllable classification across speeds |



---

## Experimental Documentation

### Experimental Setup

#### Hardware and Software Environment
All experiments were conducted on a Linux-based server equipped with:

- **Operating System**: Ubuntu 20.04 LTS  
- **CPU**: Intel Xeon Gold 6348  
- **GPU**: NVIDIA A100 40GB (CUDA 11.7)  
- **Memory**: 256 GB DDR4  
- **Framework**: TensorFlow 2.11.0 with Keras API  
- **Audio Processing Libraries**: pycochleagram, librosa, praat-parselmouth  

#### Experimental Groups
fine-tuning setups:

- **Syllable Recognition**: 84-class syllable classification  

#### Input and Output Format
- **Input**: 256×256 cochleagram images from 16 kHz Mandarin audio clips  
- **Preprocessing**: filterbank analysis, envelope compression, Lanczos interpolation  
- **Output**: Softmax probability over 63 or 84 categories (task-dependent)  

#### Test Conditions
- **Clean**: Original speech (TTS + human)    9 × 84
- **Noisy**: Augmented with THCHS-30 additive noise with SNR (1,5,10,15 dB), pitch shifting(±1,2,3), and time-stretching(0.8,0.9,1.1,1.2,1.3,1.4,1.5) 

### Data Collection and Annotation Procedures

#### Audio Sources
Two types of speech data were used for training and evaluation:

- **Synthetic Speech**:  
  Generated using Edge TTS (Microsoft Azure) engines. Both Mandarin Chinese female and male voices were used for diversity.
  Edge TTS Chinese Voices Used:

| Voice Name        | Gender | Locale    | Description                            |
|-------------------|--------|-----------|----------------------------------------|
| XiaoxiaoNeural    | Female | zh-CN     | Warm, natural – suitable for narration or news |
| XiaoyiNeural      | Female | zh-CN     | Lively – suitable for stories or cartoons |
| YunxiNeural       | Male   | zh-CN     | Professional – suitable for broadcasts |
| YunjianNeural     | Male   | zh-CN     | Energetic – suitable for sports or dialogue |
| YunyangNeural     | Male   | zh-CN     | Clear and steady – suitable for general content |
| HsiaoChenNeural   | Female | zh-TW     | Friendly and expressive – Taiwanese Mandarin |
| HsiaoYuNeural     | Female | zh-TW     | Pleasant and casual – Taiwanese Mandarin |


- **Human Recordings**:  
  A native Chinese volunteer read and recorded selected song lyrics in a quiet environment. Recordings were performed using high-quality USB microphones at a 16 kHz sampling rate.

#### Text Materials
The linguistic content used for all speech stimuli was derived from a single full-length Mandarin Chinese song. The lyrics of this song were used consistently across all conditions and speakers.

One classification task was constructed by annotating the lyrics at syllable level:

- **Syllable Task**:  
  Based on syllable-level annotations using Praat. Used to train a separate dual-pathway CNN for syllable classification + genre classification.

This setup ensured controlled comparison of syllable-level decoding while preserving consistent acoustic and lexical content.

#### Annotation Procedure for Transfer Learning

- **Class Labels**:
  - For Syllable Task: 84-class Chinese syllable labels
  Labels were encoded into one-hot vectors using a `label_map.json` file.

#### Tools and Automation

- **Praat + Parselmouth**:  
  Used for manual alignment correction and boundary visualization.

- **Edge TTS Scripting**:  
  Automatically generated WAV files from textual prompts with boundary metadata.

- **NumPy Pipelines**:  
  Converted aligned audio into cochleagrams and stored them with synchronized metadata.

### Getting Started
(Core codes is pasted below)
1. Convert audio `.mp3` files in `data/music_audio/` to 16kHz `.wav`  
2. Generate cochleagrams using `cochleagram_generator.py`  
3. Train or evaluate CNN models with `TransferLearningCNN.py`  
4. Use label mappings in `data/label/` for classification outputs  
5. Explore feature pipelines via notebooks in `modules/Demo/`  

---

### Documentation

Refer to the `docs/` folder for:
- CNN architecture diagrams  
- Experimental setup  
- Data collection and annotation procedures  

---

### Dataset Directory

This folder contains data used in the Music-EEG Cognition project, including stimulus audio files and label mappings for classification tasks.

#### music_audio/
Contains test music stimuli in MP3 format and their corresponding TextGrid annotations. These are used during the testing phase or in EEG experiments.

Each stimulus consists of:

- `musicX.mp3`: The audio file (speed-adjusted variants such as 1.0x, 2.0x, etc.)  
- `musicX.TextGrid`: Time-aligned annotations for phoneme or word-level content (Praat format)  

> Note: MP3 files must be converted to 16kHz .wav format before use in preprocessing or CNN-based models.

#### label/
Includes JSON files mapping human-readable labels (words, syllables, genres) to numeric class indices used for training and evaluating the model.

- `wordLabel.json`: Word-to-index mapping for word-level classification  
- `syllableLabel.json`: Syllable-to-index mapping for syllable-level classification  
- `genreLabel.json`: Genre-to-index mapping for music genre classification  

These files are essential for decoding model predictions and constructing categorical training labels.

---

### Usage Flow Overview

#### Audio Processing
Convert `.mp3` files in `music_audio/` to `.wav` format (16kHz) for input.

#### Label Alignment
Use `.TextGrid` files for aligning and segmenting input audio with linguistic annotations.

#### Model Training/Evaluation
Load appropriate label maps from `label/` when training or interpreting outputs from word/syllable/genre CNN classifiers.

> For more details on how these data files are used in training and evaluation, please refer to the corresponding modules under `/modules/`.

---

### Modules

This directory contains modular components used in the Music-EEG Cognition project. Each submodule performs a specific stage in the overall processing pipeline, including data preprocessing, feature extraction, transfer learning, and demo inference.

#### Submodules

#### Demo/
Includes interactive notebooks demonstrating how to process audio data, extract features, synthesize speech, and perform CNN-based predictions.

- `feature_extraction.ipynb`: Extracts acoustic features from audio stimuli.  
- `merge_features.ipynb`: Merges feature sets for model input.  
- `textToSpeech.ipynb`: Converts text into synthetic speech using TTS.  
- `demo_cnn_prediction.ipynb`: Shows prediction results from the trained CNN model.  

> Note: Audio files should be converted to 16kHz `.wav` format for compatibility with feature extraction and model input.


#### Label Extraction and Encoding
All training data were automatically labeled based on the filenames of the preprocessed cochleagram `.npy` files.

Each `.npy` file represents one stimulus segment derived from the same Chinese song, and filenames follow a structured convention such as:

```
我_1.npy, 我_2.npy, 可_1.npy, 不_1.npy
```

To generate labels for classification:

- The unique Chinese word (or syllable) was extracted from the filename prefix.
- A sorted list of all unique labels was constructed.
- Each label was assigned a unique integer ID.
- The resulting dictionary (e.g., `{ "我": 0, "可": 1, "不": 2 }`) was saved to `label_map.json`.

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



### Cochleagram Generation Code

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

- cochleagram_generator generates cochleagram images from audio signals.  
- TransferLearningCNN defines and fine-tunes the CNN model using transfer learning techniques.  

> The cochleagrams generated by `cochleagram_generator.py` should be converted into `.npy` format before used as input to the CNN defined in `TransferLearningCNN.py`.

Each module is designed to be modular and reusable across different stages of the pipeline.

---

### CNN Model Architecture and Training Configuration
We used a dual-branch convolutional neural network architecture adapted from Kell et al. (2018), consisting of:

- A shared convolutional feature extractor with three layers (conv1, conv2, conv3)
- Two task-specific classification branches: **Word** and **Genre**
- Only the Word Branch was fine-tuned to fit the Mandarin word-level and syllable-level tasks; the Genre Branch and shared layers were frozen

**Training hyperparameters**:

- Optimizer: Adam  
- Learning rate: 1e-4  
- Batch size: 32  
- Epochs: 3  
- Loss: Categorical cross-entropy  
- Loss weights: fctop_W: 1.0, fctop_G: 0.0  
- Validation split: 10%  


### Transfer Learning CNN Code

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

# Build the new model with updated word branch class count
new_num_classes_word = 63
model = build_kell2018_cnn(input_shape=(256,256,1), num_classes_word=new_num_classes_word, num_classes_genre=43)

# Load pretrained weights and assign to corresponding layers (Kell et al., 2018)
weights_early_path = "Weights/network_weights_early_layers_fixed.npy"
weights_genre_path = "Weights/network_weights_genre_branch_fixed.npy"
weights_word_path = "Weights/network_weights_word_branch_fixed.npy"

# Load weight dictionaries
weights_early = np.load(weights_early_path, allow_pickle=True).item()
weights_genre = np.load(weights_genre_path, allow_pickle=True).item()
weights_word = np.load(weights_word_path, allow_pickle=True).item()

# Combine shared and genre branch weights (these layers' shapes remain unchanged)
weights_common = {**weights_early, **weights_genre}

for layer in model.layers:
    if layer.name in weights_common:
        try:
            layer.set_weights([weights_common[layer.name]['W'], weights_common[layer.name]['b']])
            print(f"Loaded weights for {layer.name}")
        except Exception as e:
            print(f"Skipping layer {layer.name} due to shape mismatch: {e}")
            
    # Load weights for word branch layers if shape matches (fctop_W needs to be retrained)
    elif layer.name in weights_word and layer.name != "fctop_W":
        try:
            layer.set_weights([weights_word[layer.name]['W'], weights_word[layer.name]['b']])
            print(f"Loaded word branch weights for {layer.name}")
        except Exception as e:
            print(f"Skipping word branch layer {layer.name} due to shape mismatch: {e}")

# Set training strategy: freeze shared and genre layers, fine-tune only word branch
word_branch_names = {"conv4_W", "conv5_W", "pool5_W", "fc6_W", "fctop_W"}
for layer in model.layers:
    if layer.name in word_branch_names:
        layer.trainable = True
    else:
        layer.trainable = False

# Print model summary to check trainable status
model.summary()

# Load label_map and update number of word classes
label_map_path = "TrainSet/labels/wordLabel.json"
with open(label_map_path, "r", encoding="utf-8") as f:
    label2id = json.load(f)
new_num_classes_word = len(label2id)
print("Number of Chinese word classes:", new_num_classes_word)  # Should be 63

# Build model with updated word class count
model = build_kell2018_cnn(input_shape=(256,256,1), 
                           num_classes_word=new_num_classes_word, 
                           num_classes_genre=43)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss={'fctop_W': 'categorical_crossentropy', 
                    'fctop_G': 'categorical_crossentropy'},
              loss_weights={'fctop_W': 1.0, 'fctop_G': 0.0},
              metrics={'fctop_W': 'accuracy'})

# Load training data and generate labels
npy_folder = "TrainSet/cochleagrams_npy/"

X_train_list = []
y_train_list = []
for filename in os.listdir(npy_folder):
    if filename.endswith(".npy"):
        filepath = os.path.join(npy_folder, filename)
        cochleagram = np.load(filepath)
        # Expand to (256,256,1) if data is 2D
        if cochleagram.ndim == 2:
            cochleagram = np.expand_dims(cochleagram, axis=-1)
        X_train_list.append(cochleagram)
        # Extract label from filename
        base_name = os.path.splitext(filename)[0]
        if "_" in base_name:
            label = base_name.split("_")[0]
        else:
            label = base_name
        label_id = label2id[label]
        y_train_list.append(label_id)

X_train = np.array(X_train_list)
y_train = np.array(y_train_list)
# One-hot encode labels, shape (num_samples, 63)
y_train = to_categorical(y_train, num_classes=new_num_classes_word)

print("Training data X_train shape:", X_train.shape)
print("Training labels y_train shape:", y_train.shape)

# Shuffle the data
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

# Generate dummy labels for fctop_G (genre output), assuming 43 classes
dummy_y_genre = np.zeros((X_train.shape[0], 43))

# Fine-tune the model: provide both outputs
history = model.fit(X_train, {'fctop_W': y_train, 'fctop_G': dummy_y_genre},
                    batch_size=32,
                    epochs=3,
                    validation_split=0.1)
print("Fine-tuning completed!")

# Save the fine-tuned model
model_save_path = "fine_tuned_model.h5"
model.save(model_save_path)
print(f"Training complete. Model saved to {model_save_path}")
```


## Acknowledgements

This project utilizes code from [mcdermottLab/kelletal2018](https://github.com/mcdermottLab/kelletal2018), developed by Kell et al. for their research published in *Neuron* (2018). We thank the authors for making their code available.

---

## Contact

For questions or contributions, feel free to open an issue or pull request.

---

**License**: [MIT](LICENSE)  
