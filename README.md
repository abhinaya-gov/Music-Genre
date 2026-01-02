
<p align="center">
  <a href="#model-architecture">
    <img src="https://img.shields.io/badge/Architecture-View-blue?style=for-the-badge" />
  </a>
  <a href="#training-strategy">
    <img src="https://img.shields.io/badge/How%20It%20Works-Explore-green?style=for-the-badge" />
  </a>
  <a href="#future-improvements">
    <img src="https://img.shields.io/badge/Future%20Plans-Roadmap-orange?style=for-the-badge" />
  </a>
</p>

# Music Genre Classification with CNN on Mel Spectrograms

An end-to-end deep learning project that classifies music into genres using convolutional neural networks trained on mel-spectrogram representations of audio signals.

This project demonstrates the full ML pipeline: signal processing → feature extraction → model design → training → evaluation → deployment via a web interface.

---

## Project Overview

Music is a time-series signal, but CNNs operate on images. This project bridges that gap by:

1. Converting raw audio into mel spectrograms  
2. Training a VGG-style CNN on these spectrograms  
3. Using chunk-wise temporal aggregation to classify long audio tracks  
4. Deploying the trained model in a Flask web application  

Users can upload a `.wav` file and receive a predicted genre along with model confidence.

---

## Key Concepts Demonstrated

- Digital signal processing (sampling, windowing, Fourier transform)
- Mel-frequency scaling (psychoacoustically motivated features)
- CNNs for 2D pattern recognition
- Chunk-based temporal modeling
- Training deep networks from scratch
- Model serialization and inference
- Web deployment of ML models

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

## Model Architecture

### Feature Extraction

- Audio resampled to **22,050 Hz**
- Split into overlapping 10-second windows (50% overlap)
- Each window → mel spectrogram:
  - 128 mel bands
  - FFT size = 2048
  - Hop length = 512
- Spectrograms normalized per chunk

### Model Architecture (VGG-style CNN)

- Input: 1 × 128 × T mel spectrogram <br>
[Conv → ReLU → Conv → ReLU → BatchNorm → MaxPool] × 5 blocks <br>
Channels: 1 → 32 → 64 → 128 → 256 → 256 <br>
AdaptiveAvgPool → Fully Connected Layers <br>
256 → 512 → 128 → 10 (genres) <br>

<img width="920" height="572" alt="image" src="https://github.com/user-attachments/assets/d5b5b9af-a70a-459e-a886-ebfa504257b9" />


- ReLU activations
- Batch normalization for stability
- Dropout for regularization
- Cross-entropy loss
- Adam optimizer

---

## Dataset

- GTZAN Music Genre Dataset  
- 10 genres, ~100 tracks per genre  
- Each track ~30 seconds  

Each track is split into multiple overlapping windows, allowing the model to learn local time-frequency patterns while still producing a global classification.

---

## Training Strategy

- Audio split into overlapping chunks  
- Each chunk treated as an independent training example  
- During inference:
  - Model predicts genre probabilities for each chunk
  - Probabilities are averaged across all chunks
  - Final prediction = argmax of averaged probabilities  

This avoids needing RNNs or Transformers while still capturing temporal information.

---

## Web Application

A Flask web interface allows:

- Uploading `.wav` files
- Running inference on the trained model
- Displaying predicted genre + confidence
- Playing back the uploaded audio

This demonstrates practical deployment and model usability.

---

## Tech Stack

- Python
- PyTorch
- Librosa
- NumPy
- Flask
- HTML/CSS

---

## Why This Project Matters

This project is not just about training a classifier — it demonstrates:

- Understanding of signal processing, not just neural networks
- Designing architectures appropriate to the data modality
- Engineering decisions around windowing, normalization, and aggregation
- Awareness of deployment and usability
- End-to-end ownership of an ML system

It reflects real-world ML work more than a Kaggle-style notebook.

---

## Future Improvements

- Add probability distribution visualizations
- Add Grad-CAM on spectrograms for interpretability
- Experiment with pre-trained audio models (e.g. PANNs, YAMNet)
- Try Transformers on spectrogram patches
- Support more audio formats (mp3, flac)
- Add REST API endpoint

---

## Author

**Abhi**  
Interested in machine learning, deep learning, signal processing, and building models from first principles.


