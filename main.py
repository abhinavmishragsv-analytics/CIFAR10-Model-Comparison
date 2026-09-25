import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from thop import profile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# 1. DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("DEVICE:", device)
print("=" * 60)


# =========================================================
# 2. CIFAR-10 DATASET
# =========================================================

NUM_CLASSES = 10
BATCH_SIZE = 128
IMAGE_SIZE = 128


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(IMAGE_SIZE, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

from torch.utils.data import Subset

train_dataset = Subset(
    train_dataset,
    range(10000)
)

test_dataset = Subset(
    test_dataset,
    range(2000)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))


# =========================================================
# 3. MODEL 1 — PRETRAINED MOBILENET
# =========================================================

print("\nLoading pretrained MobileNetV3-Large...")

weights = MobileNet_V3_Large_Weights.DEFAULT

mobilenet = mobilenet_v3_large(weights=weights)

# Freeze feature extractor
for param in mobilenet.features.parameters():
    param.requires_grad = False

# Customize classifier for CIFAR-10
in_features = mobilenet.classifier[-1].in_features

mobilenet.classifier[-1] = nn.Linear(
    in_features,
    NUM_CLASSES
)

mobilenet = mobilenet.to(device)


# =========================================================
# 4. CUSTOM INCEPTION MODEL
# =========================================================

class InceptionBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        c1,
        c3_reduce,
        c3,
        c5_reduce,
        c5,
        pool_proj
    ):
        super().__init__()

        # 1x1 branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        # 3x3 branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c3_reduce, kernel_size=1),
            nn.BatchNorm2d(c3_reduce),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                c3_reduce,
                c3,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )

        # 5x5 branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c5_reduce, kernel_size=1),
            nn.BatchNorm2d(c5_reduce),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                c5_reduce,
                c5,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True)
        )

        # Pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Conv2d(
                in_channels,
                pool_proj,
                kernel_size=1
            ),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        return torch.cat(
            [b1, b2, b3, b4],
            dim=1
        )


class CustomInceptionNet(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        self.stem = nn.Sequential(

            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)
        )

        self.inception1 = InceptionBlock(
            64,
            32,
            32,
            64,
            16,
            32,
            32
        )

        # 32 + 64 + 32 + 32 = 160

        self.pool1 = nn.MaxPool2d(2)

        self.inception2 = InceptionBlock(
            160,
            64,
            48,
            96,
            16,
            32,
            32
        )

        # 64 + 96 + 32 + 32 = 224

        self.pool2 = nn.MaxPool2d(2)

        self.inception3 = InceptionBlock(
            224,
            96,
            64,
            128,
            32,
            64,
            64
        )

        # 96 + 128 + 64 + 64 = 352

        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.dropout = nn.Dropout(0.4)

        self.fc = nn.Linear(
            352,
            num_classes
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.inception1(x)

        x = self.pool1(x)

        x = self.inception2(x)

        x = self.pool2(x)

        x = self.inception3(x)

        x = self.global_pool(x)

        x = torch.flatten(x, 1)

        x = self.dropout(x)

        x = self.fc(x)

        return x


inception = CustomInceptionNet(
    num_classes=NUM_CLASSES
)

inception = inception.to(device)


# =========================================================
# 5. TRAINING FUNCTION
# =========================================================

def train_model(
    model,
    model_name,
    epochs=3,
    lr=0.001
):

    print("\n" + "=" * 60)
    print("TRAINING:", model_name)
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=lr
    )

    best_accuracy = 0

    for epoch in range(epochs):

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        start_time = time.time()

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        train_accuracy = (
            100 * correct / total
        )

        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {running_loss/len(train_loader):.4f} "
            f"Train Acc: {train_accuracy:.2f}% "
            f"Time: {epoch_time:.2f}s"
        )

        test_accuracy = evaluate_model(
            model,
            verbose=False
        )

        print(
            f"Test Accuracy: "
            f"{test_accuracy:.2f}%"
        )

        if test_accuracy > best_accuracy:

            best_accuracy = test_accuracy

            torch.save(
                model.state_dict(),
                f"{model_name}.pth"
            )

    return best_accuracy


# =========================================================
# 6. EVALUATION
# =========================================================

def evaluate_model(
    model,
    verbose=True
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = (
        100 * correct / total
    )

    if verbose:

        print(
            f"Test Accuracy: {accuracy:.2f}%"
        )

    return accuracy


# =========================================================
# 7. PARAMETER COUNT
# =========================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def count_all_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


# =========================================================
# 8. FLOPS
# =========================================================

def calculate_flops(model):

    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to(device)

    flops, params = profile(
        model,
        inputs=(dummy_input,),
        verbose=False
    )

    return flops, params


# =========================================================
# 9. INFERENCE TIME
# =========================================================

def measure_inference_time(
    model,
    num_iterations=100
):

    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to(device)

    # Warm-up
    with torch.no_grad():

        for _ in range(10):

            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        for _ in range(num_iterations):

            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start

    average_time = (
        total_time / num_iterations
    )

    return average_time * 1000


# =========================================================
# 10. TRAIN BOTH MODELS
# =========================================================

mobilenet_accuracy = train_model(
    mobilenet,
    "MobileNet",
    epochs=2,
    lr=0.001
)

inception_accuracy = train_model(
    inception,
    "CustomInception",
    epochs=2,
    lr=0.001
)


# =========================================================
# 11. LOAD BEST MODELS
# =========================================================

mobilenet.load_state_dict(
    torch.load(
        "MobileNet.pth",
        map_location=device
    )
)

inception.load_state_dict(
    torch.load(
        "CustomInception.pth",
        map_location=device
    )
)


# =========================================================
# 12. FINAL METRICS
# =========================================================

print("\n" + "=" * 60)
print("CALCULATING FINAL METRICS")
print("=" * 60)


# Accuracy

mobilenet_accuracy = evaluate_model(
    mobilenet
)

inception_accuracy = evaluate_model(
    inception
)


# Parameters

mobilenet_params = count_all_parameters(
    mobilenet
)

inception_params = count_all_parameters(
    inception
)


# FLOPs

print("\nCalculating MobileNet FLOPs...")

mobilenet_flops, _ = calculate_flops(
    mobilenet
)

print("Calculating Custom Inception FLOPs...")

inception_flops, _ = calculate_flops(
    inception
)


# Inference

print("\nMeasuring MobileNet inference time...")

mobilenet_inference = measure_inference_time(
    mobilenet
)

print("Measuring Custom Inception inference time...")

inception_inference = measure_inference_time(
    inception
)


# =========================================================
# 13. RESULTS TABLE
# =========================================================

results = pd.DataFrame({

    "Model": [
        "Pretrained MobileNetV3-Large",
        "Custom InceptionNet"
    ],

    "Accuracy (%)": [
        mobilenet_accuracy,
        inception_accuracy
    ],

    "Parameters": [
        mobilenet_params,
        inception_params
    ],

    "Parameters (Millions)": [
        mobilenet_params / 1e6,
        inception_params / 1e6
    ],

    "FLOPs": [
        mobilenet_flops,
        inception_flops
    ],

    "GFLOPs": [
        mobilenet_flops / 1e9,
        inception_flops / 1e9
    ],

    "Inference Time (ms)": [
        mobilenet_inference,
        inception_inference
    ]
})


print("\n")
print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    results.to_string(
        index=False
    )
)


# Save CSV

results.to_csv(
    "results.csv",
    index=False
)


# =========================================================
# 14. GRAPH 1 — ACCURACY
# =========================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=results,
    x="Model",
    y="Accuracy (%)"
)

plt.title(
    "Accuracy Comparison"
)

plt.ylabel(
    "Test Accuracy (%)"
)

plt.xlabel(
    "Model"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    "accuracy_comparison.png",
    dpi=300
)

plt.show()


# =========================================================
# 15. GRAPH 2 — PARAMETERS
# =========================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=results,
    x="Model",
    y="Parameters (Millions)"
)

plt.title(
    "Trainable Model Size Comparison"
)

plt.ylabel(
    "Parameters (Millions)"
)

plt.xlabel(
    "Model"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    "parameters_comparison.png",
    dpi=300
)

plt.show()


# =========================================================
# 16. GRAPH 3 — FLOPS
# =========================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=results,
    x="Model",
    y="GFLOPs"
)

plt.title(
    "Computational Complexity Comparison"
)

plt.ylabel(
    "GFLOPs"
)

plt.xlabel(
    "Model"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    "flops_comparison.png",
    dpi=300
)

plt.show()


# =========================================================
# 17. GRAPH 4 — INFERENCE TIME
# =========================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=results,
    x="Model",
    y="Inference Time (ms)"
)

plt.title(
    "Inference Time Comparison"
)

plt.ylabel(
    "Inference Time (ms)"
)

plt.xlabel(
    "Model"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    "inference_comparison.png",
    dpi=300
)

plt.show()


# =========================================================
# 18. COMBINED COMPARISON
# =========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9)
)


# Accuracy

axes[0, 0].bar(
    results["Model"],
    results["Accuracy (%)"]
)

axes[0, 0].set_title(
    "Accuracy"
)

axes[0, 0].set_ylabel(
    "Accuracy (%)"
)


# Parameters

axes[0, 1].bar(
    results["Model"],
    results["Parameters (Millions)"]
)

axes[0, 1].set_title(
    "Parameters"
)

axes[0, 1].set_ylabel(
    "Millions"
)


# FLOPs

axes[1, 0].bar(
    results["Model"],
    results["GFLOPs"]
)

axes[1, 0].set_title(
    "FLOPs"
)

axes[1, 0].set_ylabel(
    "GFLOPs"
)


# Inference

axes[1, 1].bar(
    results["Model"],
    results["Inference Time (ms)"]
)

axes[1, 1].set_title(
    "Inference Time"
)

axes[1, 1].set_ylabel(
    "Milliseconds"
)


for ax in axes.flat:

    ax.tick_params(
        axis="x",
        rotation=20
    )


plt.suptitle(
    "CIFAR-10 Model Comparison",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "complete_comparison.png",
    dpi=300
)

plt.show()


print("\nAll results saved successfully!")