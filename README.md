# Semantic Segmentation with CamVid Dataset

## Introduction

Semantic segmentation is a core task in computer vision, involving the classification of each pixel in an image into predefined categories. In this project, we tackled semantic segmentation using the CamVid dataset, inspired by the foundational work of Long et al. [[1]](#references) and educational materials from Khalequzzaman Likhon’s course [[2]](#references).

We implemented three different semantic segmentation models:
- A Fully Convolutional Network (FCN) with a ResNet-50 backbone and custom decoder,
- A U-Net architecture enhanced with an attention mechanism,
- The DeepLabv3+ architecture [[3]](#references).

---

## Dataset and Preprocessing

We used the **CamVid (Cambridge-driving Labeled Video Database)** dataset, which includes images annotated for 32 semantic classes. The dataset contains:

- `train/`, `val/`, `test/` image folders
- Corresponding label folders
- A CSV file mapping class names to RGB values

We used this CSV to convert between RGB and class masks. The following data transformations were applied:

### Training Data Augmentation
- `Resize(400, 520)`
- `RandomCrop(352, 480)`
- `HorizontalFlip` (p=0.5)
- `Rotate` (±15°)
- `GaussianBlur` (kernel size 3–5, p=0.3)
- `ColorJitter`
- `Normalize(mean=(0.390, 0.405, 0.414), std=(0.274, 0.285, 0.297))`
- `ToTensorV2`

### Validation & Test Transformations
- `Resize(352, 480)`
- `Normalize` (same as above)
- `ToTensorV2`

---

## Model Architectures

### 1. Fully Convolutional Network (FCN)
- **Encoder:** ResNet-50 (pretrained)
- **Decoder:** Multiple ConvTranspose layers with BatchNorm and ReLU
- **Final Output:** Upsampled to (352, 480), with 32 class channels

### 2. U-Net with Attention Mechanism
- **Encoder:** ResNet-50 (pretrained), with feature maps e1–e5
- **Decoder:** Attention gates + decoder blocks at each scale
- **Final Output:** Upsampled to (352, 480), 32 class channels

### 3. DeepLabv3+
- **Encoder:** ResNet-50 (pretrained)
- **ASPP Module:** Multi-scale context with atrous convolutions
- **Decoder:** Combines low-level features with ASPP output
- **Final Output:** Upsampled to (352, 480), 32 class channels

---

## Training Setup

### Hyperparameters
- **Optimizer:** Adam
- **Learning Rate:** 1e-4
- **Weight Decay:** 
  - FCN: 1e-4
  - U-Net & DeepLabv3+: 1e-5
- **Batch Size:** 4 (due to GPU memory limits)
- **LR Scheduler:** `ReduceLROnPlateau` (factor=0.1, patience=3)
- **Early Stopping:** Patience = 7 epochs, min delta = 0.001

### Loss Function

A weighted combination of Dice Loss and Jaccard Loss:

\[
\text{Dice Loss} = 1 - \frac{2 \sum p_i g_i + \epsilon}{\sum p_i + \sum g_i + \epsilon}
\]

\[
\text{Jaccard Loss} = 1 - \frac{\sum p_i g_i + \epsilon}{\sum p_i + \sum g_i - \sum p_i g_i + \epsilon}
\]

\[
\text{Combined Loss} = 0.7 \cdot \text{Dice Loss} + 0.3 \cdot \text{Jaccard Loss}
\]

---

## Team Members

- Mehul Dinesh Jain  
- Meryem Hanyn  
- Nurettin Berke Çevik  
- Bora Özdamar  
- Daniel Alexander Meija Romero  
- Ghazaleh Alizadehbirjandi

---

## References

1. Long, J., Shelhamer, E., & Darrell, T. (2015). [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038)  
2. [Semantic Segmentation in PyTorch – GitHub Notebook](https://github.com/khalequzzamanlikhon/DeepLearning-ComputerVision/blob/master/08-Segmentation-Detection/01-Semantic-Segmentation.ipynb)  
3. Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLabv3+)](https://arxiv.org/pdf/1802.02611)  
4. [Kaggle Notebook – Semantic Segmentation PyTorch from Scratch](https://www.kaggle.com/code/likhon148/semantic-segmentation-pytorch-scratch)

