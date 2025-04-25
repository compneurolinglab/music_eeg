# Music-EEG-Cognition

This project explores the neural channel capacity of the human brain in processing music and speech using EEG data and deep learning models.

Inspired by previous studies in auditory cognition and information theory, we investigate how the brain allocates resources when decoding musical genres versus linguistic content under varying playback conditions. The framework integrates EEG-based experiments, linguistic feature analysis, and a dual-pathway CNN model to simulate auditory perception and establish parallels between biological and computational processing limits.

---

## Project Structure

Music-EEG-Cognition/ │ ├── data/ # Contains test audio stimuli and label mappings │ ├── music_audio/ # MP3 music files and TextGrid annotations (test stimuli) │ └── label/ # JSON mappings for word, syllable, and genre classification │ ├── modules/ # Model and preprocessing code │ ├── Demo/ # Feature extraction, TTS, and CNN demo notebooks │ └── cnn_transfer/ # Transfer learning CNN and cochleagram generator │ ├── docs/ # Project documentation, architecture diagrams │ └── README.md # Project description and overview


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

## Contact

For questions or contributions, feel free to open an issue or pull request.
qimiaogao2-c@my.cityu.edu.hk

---

**License**: [MIT](LICENSE)  
**Maintainer**: [@CiaraGao](https://github.com/CiaraGao)
