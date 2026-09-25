# CIFAR-10 Model Comparison

<p align="center">

**Pretrained MobileNetV3-Large vs Custom InceptionNet**

A PyTorch-based deep learning experiment comparing a customized pretrained CNN with a newly developed Inception-style architecture on the CIFAR-10 image classification dataset.

</p>

---

## 📌 Problem Statement

**Problem Number: 5**

The objective of this laboratory experiment is to:

1. Customize an existing pretrained deep learning model.
2. Develop a new CNN architecture.
3. Train both models on the same classification task.
4. Compare their performance using:
   - Accuracy
   - Number of Parameters / Weights
   - FLOPs
   - Inference Time

### Selected Configuration

| Component | Selection |
|---|---|
| Dataset | CIFAR-10 |
| Framework | PyTorch |
| Pretrained Model | MobileNetV3-Large |
| Custom Model | Custom InceptionNet |
| Task | Image Classification |
| Number of Classes | 10 |

---

## 🧠 Approach

Two different modeling approaches were evaluated.

### 1. Pretrained MobileNetV3-Large

A MobileNetV3-Large model pretrained on ImageNet was used as the baseline transfer-learning model.

The original classification layer was replaced with a new fully connected layer containing **10 output neurons**, corresponding to the ten CIFAR-10 classes.

The pretrained feature extractor was frozen while the customized classifier was trained for the CIFAR-10 task.

**Pipeline:**

```text
ImageNet Pretrained MobileNetV3-Large
                ↓
       Freeze Feature Extractor
                ↓
       Replace Final Classifier
                ↓
       10-Class CIFAR-10 Output
                ↓
              Train
