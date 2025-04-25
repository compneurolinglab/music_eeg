# Data Collection and Annotation Procedures

## Audio Sources

Two types of speech data were used for training and 10% of them were used for validation:

- **Synthetic Speech**:  
  Generated using **Edge TTS** (Microsoft Azure) engines. Both Mandarin Chinese female and male voices were used for diversity.
  
- **Human Recordings**:  
  Chinese volunteers read and recorded selected song lyrics in a quiet environment. Recordings were performed using high-quality USB microphones at a 16 kHz sampling rate.

## Text Materials

The linguistic content used for all speech stimuli was derived from a **single full-length Mandarin Chinese song**. The lyrics of this song were used consistently across all conditions and speakers.

Two classification tasks were constructed by annotating the lyrics at different linguistic levels:

- **Word Task**:  
  Based on word-level annotations using Praat. Used to train a dual-pathway CNN for **word classification** + **genre classification**.

- **Syllable Task**:  
  Based on syllable-level annotations using Praat. Used to train a separate dual-pathway CNN for **syllable classification** + **genre classification**.

This setup ensured controlled comparison of word- and syllable-level decoding while preserving consistent acoustic and lexical content.

## Annotation Procedure

Each audio clip was segmented and labeled with the following information:

- **Syllable Boundaries**:  
  Time-aligned using forced alignment scripts and manually corrected using **Praat**.
  
- **Word Onsets**:  
  Estimated from the original text structure and verified against the aligned audio in **Praat**.
  
- **Class Labels**:  
  - For **Word Task**: 63-class Chinese word labels  
  - For **Syllable Task**: 84-class Chinese syllable labels  
  Labels were encoded into one-hot vectors using a `label_map.json` file.

## Tools and Automation

- **Praat + Parselmouth**:  
  Used for manual alignment correction and boundary visualization.
  
- **Edge TTS Scripting**:  
  Automatically generated WAV files from textual prompts with boundary metadata.
  
- **NumPy Pipelines**:  
  Converted aligned audio into cochleagrams and stored them with synchronized metadata.

## Label Extraction and Encoding

All training data were automatically labeled based on the filenames of the preprocessed cochleagram `.npy` files.

Each `.npy` file represents one stimulus segment derived from the same Chinese song, and filenames follow a structured convention such as:

```
我_1.npy, 我_2.npy, 我_3.npy, 可以.npy, 不拘一格.npy
```

To generate labels for classification:

- The unique Chinese word (or syllable) was extracted from the filename prefix.
- A sorted list of all unique labels was constructed.
- Each label was assigned a unique integer ID.
- The resulting dictionary (e.g., `{ "我": 0, "可以": 1, "不拘一格": 2 }`) was saved to `label_map.json`.

Python script example:

```python
import os
import json

npy_folder = "autodl-tmp/TrainDataSet/cochleagrams_npy"
npy_files = [f.replace('.npy', '') for f in os.listdir(npy_folder) if f.endswith('.npy')]
label_dict = {word: idx for idx, word in enumerate(sorted(npy_files))}

with open("autodl-tmp/TrainDataSet/labels/label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_dict, f, ensure_ascii=False, indent=4)

print("successfully generated")
```

This label map was used for one-hot encoding in both the word and syllable classification tasks.
