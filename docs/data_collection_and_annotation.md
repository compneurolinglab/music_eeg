# Data Collection and Annotation Procedures

## Audio Sources

Two types of speech data were used for training and evaluation:

- **Synthetic Speech**:  
  Generated using **Edge TTS** (Microsoft Azure) and **Google TTS** engines. Both Mandarin Chinese female and male voices were used for diversity.
  
- **Human Recordings**:  
  A set of Chinese volunteers read and recorded selected song lyrics in a quiet environment. Recordings were performed using high-quality USB microphones at a 16 kHz sampling rate.

## Text Materials

The linguistic content used for audio generation included:

- **Song Lyrics**:  
  Over 10,000 unique lines of Chinese lyrics sourced from public-domain songs, curated to ensure phonetic coverage and word diversity.
  
- **Synthetic Sentences**:  
  Additional Mandarin sentences were created using prompt templates to enrich low-frequency word coverage and syllable transitions.

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

## Annotation Quality Control

- **Annotators**:  
  3 bilingual Mandarin speakers with linguistic training were involved in reviewing boundaries.
  
- **Validation Process**:  
  Each segment was reviewed by two annotators. In case of mismatch, a third annotator resolved the conflict. Final boundary error tolerance was controlled within ±20 ms.
