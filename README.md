# Music-Genre
This simple CNN model predicts the genre of music based on audios and their mel-spectograms.

# CNN-Based Music Genre Classification using Mel Spectrograms

This repository contains a **from-scratch implementation of a Convolutional Neural Network (CNN)** for **music genre classification** using **mel-spectrogram images**.  
The project explores how visual deep learning models can be applied to audio by transforming sound into a time–frequency representation.

Rather than relying on pretrained audio models, this implementation focuses on **building and training a CNN end to end** to understand the full pipeline.

---

## Problem Statement

Music genre classification aims to automatically assign a genre label (e.g., rock, jazz, classical) to an audio track.  
Since raw audio is difficult for standard CNNs to process directly, the audio is converted into **mel-spectrograms**, which encode frequency intensity over time and can be treated as images.

---

## Model Architecture

The model is inspired by **VGG-style CNNs** and consists of multiple convolutional blocks followed by a fully connected classifier.

Each convolutional block includes:
- Convolution layers with small kernels
- ReLU activations
- Batch Normalization
- Max Pooling for spatial downsampling

The classifier head uses:
- Fully connected layers
- Dropout for regularization
- Softmax output over **10 genre classes**

**Input:** `224 × 224` mel-spectrogram images  
**Output:** Probability distribution over genres

---

## Data Preprocessing

Before being passed to the model, each spectrogram image undergoes:

- **Automatic border removal** using a custom `RemoveBorder` transform  
  (to remove uniform padding introduced during spectrogram generation)
- Resizing to a fixed resolution
- Conversion to PyTorch tensors

This preprocessing ensures consistent inputs and reduces noise in the training data.

---

## Dataset & Splitting

- Data is loaded using `torchvision.datasets.ImageFolder`
- Genre labels are inferred from folder names
- The dataset is manually split into:
  - Training set
  - Development (validation) set
  - Test set

This separation allows performance to be monitored during training and evaluated on unseen data.

---

## Training Setup

- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Batch Size: 32
- Training Device:
  - Apple Silicon GPU (MPS) if available
  - CPU otherwise

Training accuracy and validation accuracy are tracked across epochs to monitor learning behavior and generalization.

---

## Evaluation & Results

- Final evaluation is performed on a held-out test set
- Accuracy is reported for:
  - Training set
  - Development set
  - Test set
- Training loss is visualized across epochs to inspect convergence

The results provide insight into how well a CNN can learn genre-specific patterns from mel-spectrograms.

---

## Key Takeaways

- CNNs can effectively learn from audio when represented as spectrograms
- Proper preprocessing significantly impacts model performance
- Genre classification is challenging due to overlapping musical characteristics
- Model complexity must be balanced against dataset size

---

## Possible Improvements

- Replace VGG-style CNN with ResNet or EfficientNet
- Add temporal modeling using CRNN architectures
- Apply spectrogram-specific data augmentation
- Experiment with longer audio segments and aggregation strategies

---

## Disclaimer

This project is intended for **learning and experimentation**.  
It prioritizes understanding the full training pipeline over achieving state-of-the-art performance.

---

