
<p align="center">
  <a href="#-model-architecture">
    <img src="https://img.shields.io/badge/Architecture-View-blue?style=for-the-badge" />
  </a>
  <a href="#-how-it-works">
    <img src="https://img.shields.io/badge/How%20It%20Works-Explore-green?style=for-the-badge" />
  </a>
  <a href="#-possible-improvements">
    <img src="https://img.shields.io/badge/Future%20Plans-Roadmap-orange?style=for-the-badge" />
  </a>
</p>


# 🎵 Music Genre Classification using CNNs on Mel-Spectrograms

This repository contains a **from-scratch PyTorch implementation of a Convolutional Neural Network (CNN)** for **music genre classification** using **mel-spectrogram representations of audio**.

Instead of relying on pretrained audio models, this project focuses on building the entire pipeline end-to-end:
**audio → signal processing → spectrograms → CNN → evaluation → inference**.

The goal is to understand how visual deep learning models can be applied to audio through time-frequency representations.

---

## Why This Project?

This project was built to:

- Understand how audio can be transformed into image-like representations for deep learning.
- Learn the full ML pipeline: data preprocessing, modeling, training, evaluation, and inference.
- Experiment with CNNs on non-visual data.
- Explore trade-offs between model complexity, dataset size, and performance.
- Build a system that works on raw `.wav` files rather than curated image datasets.

It is intentionally learning-focused rather than benchmark-driven.

---

## Problem Statement

Music genre classification aims to automatically assign a genre label (e.g., rock, jazz, classical) to an audio track.

Raw audio waveforms are difficult for CNNs to process directly, so each audio file is converted into **mel-spectrograms**, which encode frequency intensity over time and can be treated as images.

The model learns genre-specific spectral patterns from these representations.

---

## Pipeline Overview

```
Audio (.wav)
   ↓
Split into overlapping chunks (10s, 50% overlap)
   ↓
Convert each chunk → Mel-spectrogram
   ↓
Normalize spectrograms
   ↓
CNN predicts genre probabilities per chunk
   ↓
Average probabilities across chunks
   ↓
Final genre prediction
```

---

## Data Preprocessing

### Audio Loading

- Audio is loaded using `librosa.load` at **22,050 Hz**
- Converted to mono and normalized automatically

### Chunking

Each audio file is split into overlapping windows:

- Window size: **10 seconds**
- Overlap: **50%**

This increases the number of training samples and helps capture temporal variation.

### Spectrogram Generation

Each chunk is converted into a mel-spectrogram:

- `n_mels = 128`
- `n_fft = 2048`
- `hop_length = 512`
- Converted to decibel scale and normalized per chunk

This produces a `(128 × time)` representation per chunk.

---

## Dataset

- The dataset is organized as:

```
root/
 ├── blues/
 ├── classical/
 ├── jazz/
 ├── metal/
 └── ...
```

- Loaded using `torchvision.datasets.ImageFolder` logic applied to generated spectrograms
- Labels are inferred from folder names
- Data is split using `train_test_split` with stratification (80% train / 20% validation)

---

## Model Architecture

The model is inspired by **VGG-style CNNs** and implemented from scratch.

### Architecture

- 5 convolutional blocks:
  - Conv → ReLU → Conv → ReLU → BatchNorm → MaxPool
- Adaptive average pooling to remove dependency on input width
- Fully connected classifier:
  - 256 → 512 → 128 → 10
  - Dropout for regularization

**Input:** `(1 × 128 × T)` mel-spectrogram  
**Output:** Probability distribution over 10 genres

---

## Training Setup

- Loss: `CrossEntropyLoss`
- Optimizer: Adam (`lr = 0.001`)
- Batch size: 32
- Device:
  - Apple Silicon (MPS) if available
  - CPU otherwise
- Epochs: 50

Training and validation accuracy and loss are tracked across epochs.

---

## Inference

For inference on full audio files:

1. Audio is split into chunks.
2. Each chunk is classified independently.
3. Softmax probabilities are averaged across chunks.
4. Final prediction is the class with highest mean probability.

This improves robustness compared to classifying only a single segment.

---

## Evaluation

- Performance is evaluated on a held-out validation set.
- Metrics:
  - Accuracy
  - Loss curves
- Confusion analysis can be added later.

---

## Key Takeaways

- CNNs can learn meaningful audio features when audio is represented as spectrograms.
- Preprocessing choices (chunking, normalization, resolution) strongly affect performance.
- Genre classification is inherently difficult due to overlapping musical characteristics.
- Simpler models often generalize better on small datasets.

---

## Possible Improvements

- Use CRNN or temporal attention models.
- Add spectrogram-specific augmentation (time/frequency masking).
- Experiment with pretrained backbones.
- Increase dataset size and diversity.
- Add real-time audio input inference.

---

## Disclaimer

This project is intended for learning and experimentation.
It prioritizes clarity, modularity, and understanding over state-of-the-art performance.

---

## Author

**Abhi** — Learning-focused ML, DL & signal processing projects.
