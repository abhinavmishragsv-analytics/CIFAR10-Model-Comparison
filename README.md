\# CIFAR-10 Model Comparison



PyTorch implementation comparing a customized pretrained MobileNetV3-Large with a custom Inception-style CNN on CIFAR-10.



\## Problem



\- Problem Number: 5

\- Dataset: CIFAR-10

\- Framework: PyTorch

\- Pretrained Model: MobileNetV3-Large

\- Custom Model: Custom InceptionNet



\## Experimental Setup



| Parameter | Value |

|---|---|

| Input Size | 64 × 64 |

| Batch Size | 128 |

| Epochs | 2 |

| Training Samples | 10,000 |

| Testing Samples | 2,000 |

| Device | CPU |

| Classes | 10 |



\## Results



| Model | Accuracy (%) | Parameters | GFLOPs | Inference Time (ms) |

|---|---:|---:|---:|---:|

| Pretrained MobileNetV3-Large | 77.05 | 4,214,842 | 0.0781 | 11.454 |

| Custom InceptionNet | 39.75 | 308,826 | 0.3234 | 21.278 |



\## Comparison



MobileNetV3-Large achieved 77.05% accuracy, while the Custom InceptionNet achieved 39.75%.



The Custom InceptionNet has substantially fewer parameters, but its measured FLOPs and CPU inference time are higher.



This demonstrates that parameter count alone does not determine computational complexity or inference performance.



\## Graphs



\### Accuracy



!\[Accuracy](accuracy\_comparison.png)



\### Parameters



!\[Parameters](parameters\_comparison.png)



\### FLOPs



!\[FLOPs](flops\_comparison.png)



\### Inference Time



!\[Inference Time](inference\_comparison.png)



\### Overall Comparison



!\[Overall Comparison](complete\_comparison.png)



\## Technologies



\- Python

\- PyTorch

\- Torchvision

\- THOP

\- Pandas

\- Matplotlib

\- Seaborn

