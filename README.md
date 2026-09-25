# CIFAR-10 Model Comparison
### Pretrained MobileNetV3-Large vs Custom InceptionNet

A PyTorch-based deep learning experiment comparing a customized pretrained **MobileNetV3-Large** model with a newly developed **Custom InceptionNet** for image classification on the **CIFAR-10** dataset.

---

## 1. Problem Statement

The objective of this laboratory experiment is to customize an existing pretrained deep learning model, develop a new model, and compare both models in terms of:

- Classification Accuracy
- Number of Parameters / Weights
- FLOPs
- Inference Time

### Selected Problem

| Parameter | Value |
|---|---|
| Problem Number | 5 |
| Dataset | CIFAR-10 |
| Framework | PyTorch |
| Pretrained Model | MobileNetV3-Large |
| Custom Model | Custom InceptionNet |
| Task | Image Classification |
| Number of Classes | 10 |

---

## 2. Objectives

The main objectives of this experiment are:

1. To understand transfer learning using a pretrained CNN.
2. To customize MobileNetV3-Large for CIFAR-10 classification.
3. To develop an Inception-style CNN from scratch.
4. To train and evaluate both models.
5. To compare their accuracy.
6. To calculate the number of model parameters.
7. To calculate FLOPs.
8. To measure inference time.
9. To visualize and analyze the performance differences.

---

## 3. Dataset — CIFAR-10

CIFAR-10 is a standard image classification dataset containing RGB images belonging to 10 different classes.

The original images have a resolution of:

```text
32 × 32 × 3

The ten classes are:

Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck

For this experiment, the images were resized to:

64 × 64 × 3

Due to computational and time limitations, a subset of the dataset was used.

Dataset Component	Samples Used
Training Samples	10,000
Testing Samples	2,000
Number of Classes	10
Image Type	RGB
Original Resolution	32 × 32
Input Resolution	64 × 64
4. Methodology

The overall workflow of the experiment was:

                    CIFAR-10 Dataset
                           |
                           v
                  Data Preprocessing
                           |
                           v
                    Resize to 64x64
                           |
              +------------+------------+
              |                         |
              v                         v
     Pretrained MobileNet       Custom InceptionNet
              |                         |
              v                         v
      Customize Classifier       Train From Scratch
              |                         |
              +------------+------------+
                           |
                           v
                    Model Evaluation
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
       Accuracy       Parameters          FLOPs
                           |
                           v
                    Inference Time
                           |
                           v
                 Graphical Comparison
5. Model 1 — Pretrained MobileNetV3-Large
5.1 Overview

MobileNetV3-Large is a convolutional neural network designed to provide efficient image classification with relatively low computational requirements.

In this experiment, MobileNetV3-Large was initialized using pretrained ImageNet weights.

The original classification layer was then customized for the CIFAR-10 classification task.

5.2 Customization

The original MobileNetV3-Large classifier was designed for ImageNet classification.

Since CIFAR-10 contains only 10 classes, the final classification layer was replaced with a new fully connected layer containing 10 output neurons.

The feature extraction layers were frozen during training.

The process was:

ImageNet Pretrained MobileNetV3-Large
                |
                v
        Load Pretrained Weights
                |
                v
        Freeze Feature Extractor
                |
                v
        Replace Final Classifier
                |
                v
          10-Class Output
                |
                v
          Train on CIFAR-10

The important customization was:

weights = MobileNet_V3_Large_Weights.DEFAULT

mobilenet = mobilenet_v3_large(
    weights=weights
)

for param in mobilenet.features.parameters():
    param.requires_grad = False

in_features = mobilenet.classifier[-1].in_features

mobilenet.classifier[-1] = nn.Linear(
    in_features,
    10
)

This represents the transfer-learning component of the experiment.

6. Model 2 — Custom InceptionNet
6.1 Overview

The second model was developed from scratch using the basic concept of the Inception architecture.

Instead of applying only one type of convolution to the input, the model processes the input through multiple parallel branches.

The branches used were:

1 × 1 convolution
3 × 3 convolution
5 × 5 convolution
Max pooling followed by 1 × 1 convolution

The outputs of all branches are concatenated.

6.2 Inception Block

The architecture can be represented as:

                       +--- 1x1 Convolution --------+
                       |                            |
                       +--- 1x1 -> 3x3 Convolution-+
                       |                            |
Input -----------------+--- 1x1 -> 5x5 Convolution-+---> Concatenate
                       |                            |
                       +--- MaxPool -> 1x1 --------+

This allows the network to extract features at different spatial scales.

The custom network consists of multiple lightweight Inception blocks followed by:

Inception Blocks
      |
      v
Global Average Pooling
      |
      v
Dropout
      |
      v
Fully Connected Layer
      |
      v
10 Classes
7. Experimental Configuration
Parameter	Value
Framework	PyTorch
Device	CPU
Dataset	CIFAR-10
Training Samples	10,000
Testing Samples	2,000
Input Resolution	64 × 64
Batch Size	128
Training Epochs	2
Number of Classes	10
Optimizer	Adam
Loss Function	Cross-Entropy Loss
8. Evaluation Metrics

The two models were compared using four metrics.

8.1 Accuracy

Accuracy measures the percentage of correctly classified test samples.

Accuracy =
Correct Predictions / Total Predictions × 100
8.2 Parameters / Weights

Parameters represent the learnable values contained within the neural network.

The parameter count gives an indication of the model size and memory requirements.

8.3 FLOPs

FLOPs stands for Floating Point Operations.

It estimates the amount of computational work required for a forward pass through the network.

The results are reported in both FLOPs and GFLOPs.

1 GFLOP = 10^9 FLOPs
8.4 Inference Time

Inference time measures the time required by the trained model to perform a forward pass and generate a prediction.

The measurement was performed on the CPU after warm-up iterations.

Inference time is hardware-dependent and can vary between systems.

9. Experimental Results

The final results obtained from the experiment are:

Model	Accuracy (%)	Parameters	Parameters (M)	FLOPs	GFLOPs	Inference Time (ms)
Pretrained MobileNetV3-Large	77.05	4,214,842	4.215	78,122,776	0.0781	11.454
Custom InceptionNet	39.75	308,826	0.309	323,383,072	0.3234	21.278
10. Accuracy Analysis

The pretrained MobileNetV3-Large achieved:

77.05%

test accuracy.

The Custom InceptionNet achieved:

39.75%

test accuracy.

The difference between the models was:

77.05 - 39.75 = 37.30 percentage points

The higher accuracy of MobileNetV3-Large can be partly attributed to its ImageNet-pretrained feature representations.

In contrast, the Custom InceptionNet was initialized from scratch and trained for only two epochs using a subset of CIFAR-10.

Therefore, these results represent the specific experimental configuration and should not be interpreted as fully converged benchmark results.

11. Parameter Analysis

The parameter counts were:

Model	Parameters
MobileNetV3-Large	4,214,842
Custom InceptionNet	308,826

The Custom InceptionNet contains approximately:

4,214,842 / 308,826 ≈ 13.65

times fewer parameters than MobileNetV3-Large.

Therefore, the Custom InceptionNet has a substantially smaller parameter footprint.

However, parameter count alone does not determine computational complexity.

12. FLOPs Analysis

The measured computational complexity was:

Model	GFLOPs
MobileNetV3-Large	0.0781
Custom InceptionNet	0.3234

The Custom InceptionNet required approximately:

0.3234 / 0.0781 ≈ 4.14

times more FLOPs than MobileNetV3-Large for the measured input configuration.

The higher computational cost is related to the multiple convolutional branches used by the Inception-style architecture, particularly the 3 × 3 and 5 × 5 convolutions.

13. Inference Time Analysis

The measured inference times were:

Model	Inference Time
MobileNetV3-Large	11.454 ms
Custom InceptionNet	21.278 ms

The Custom InceptionNet required approximately:

21.278 / 11.454 ≈ 1.86

times the measured inference time of MobileNetV3-Large.

This measurement was performed on the CPU used during the experiment and may vary on different hardware.

14. Key Observation

One of the most important observations from this experiment is:

A model with fewer parameters does not necessarily require fewer computational operations or provide faster inference.

The Custom InceptionNet has approximately 13.6× fewer parameters than MobileNetV3-Large.

However:

It requires approximately 4.14× more FLOPs.
It has approximately 1.86× higher measured inference time.
It achieved lower test accuracy under the experimental configuration.

This demonstrates why multiple metrics should be considered when evaluating deep learning architectures.

15. Visualizations
Accuracy Comparison

Parameter Count Comparison

FLOPs Comparison

Inference Time Comparison

Overall Model Comparison

16. Comparative Summary
Metric	MobileNetV3-Large	Custom InceptionNet	Observation
Accuracy	77.05%	39.75%	MobileNet achieved higher accuracy
Parameters	4.215 M	0.309 M	Custom InceptionNet has fewer parameters
GFLOPs	0.0781	0.3234	Custom InceptionNet requires more computation
Inference Time	11.454 ms	21.278 ms	MobileNet had lower measured CPU inference time

The results demonstrate a trade-off between model size, computational complexity and classification performance.

17. Limitations

The experiment was performed under limited computational resources.

The main limitations were:

Only 10,000 training samples were used.
Only 2,000 test samples were used.
Both models were trained for only two epochs.
Training and inference were performed on CPU.
The original CIFAR-10 images were resized from 32 × 32 to 64 × 64.
The Custom InceptionNet was trained from scratch.
Inference time is dependent on the hardware and software environment.

Therefore, the reported accuracy values should be interpreted as results for the specified laboratory configuration rather than fully optimized benchmark results.

18. Technologies Used
Technology	Purpose
Python	Programming Language
PyTorch	Deep Learning Framework
Torchvision	Dataset and Pretrained Models
THOP	FLOPs Calculation
Pandas	Result Processing
NumPy	Numerical Operations
Matplotlib	Data Visualization
Seaborn	Visualization
19. Project Structure
CIFAR10-Model-Comparison/
│
├── main.py
├── README.md
├── results.csv
├── .gitignore
│
├── accuracy_comparison.png
├── parameters_comparison.png
├── flops_comparison.png
├── inference_comparison.png
└── complete_comparison.png
20. How to Run
Clone the Repository
git clone https://github.com/YOUR_USERNAME/CIFAR10-Model-Comparison.git
cd CIFAR10-Model-Comparison
Install Dependencies
pip install torch torchvision torchaudio
pip install thop pandas matplotlib seaborn torchinfo
Run the Experiment
python main.py

The program downloads the CIFAR-10 dataset, performs preprocessing, trains both models, evaluates their performance and generates the comparison graphs.

21. Conclusion

This laboratory experiment compared a customized pretrained MobileNetV3-Large model with a newly developed Custom InceptionNet for CIFAR-10 image classification using PyTorch.

Under the experimental configuration, MobileNetV3-Large achieved 77.05% test accuracy, while the Custom InceptionNet achieved 39.75%.

MobileNetV3-Large contained approximately 4.215 million parameters, whereas the Custom InceptionNet contained approximately 0.309 million parameters.

Despite having significantly fewer parameters, the Custom InceptionNet required more computational operations, with 0.3234 GFLOPs compared with 0.0781 GFLOPs for MobileNetV3-Large.

The measured CPU inference time was 11.454 ms for MobileNetV3-Large and 21.278 ms for the Custom InceptionNet.

The experiment demonstrates that deep learning models should be evaluated using multiple metrics rather than relying on a single measure. Accuracy, parameter count, computational complexity and inference efficiency can provide different perspectives on the practical characteristics of a model.

22. References
PyTorch
Torchvision
CIFAR-10 Dataset
Howard et al., Searching for MobileNetV3
Szegedy et al., Going Deeper with Convolutions
Author

Abhinav Mishra
B.Tech — Artificial Intelligence & Data Science
Gati Shakti Vishwavidyalaya
