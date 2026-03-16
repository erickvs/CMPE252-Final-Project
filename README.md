# Comparative Analysis of Classical Machine Learning and Deep Learning for Image Classification on CIFAR-10

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)

## 📌 Executive Summary
This project aims to classify images from the CIFAR-10 dataset using three distinct paradigms:
1.  **Classical Machine Learning:** Dimensionality reduction via PCA followed by an optimized Linear SVM.
2.  **Deep Learning (From Scratch):** A modified ResNet-18 CNN architecture trained natively on 32x32 images.
3.  **Foundation Model (Transfer Learning):** A Vision Transformer (ViT-B/16) pre-trained on ImageNet.

We aim to evaluate and quantify the "cost" of accuracy across these paradigms in terms of training time and resource usage.

## 💻 Hardware Note
*Developed on an Apple M4 Max (128GB Unified Memory) using PyTorch MPS.* 
The codebase features a dynamic hardware router that will automatically target `cuda` on NVIDIA machines, `mps` on Macs, and gracefully fallback to `cpu`. You can run this anywhere.

## 🚀 Quickstart

1.  **Clone and enter directory:**
    ```bash
    git clone https://github.com/erickvs/cmpe252_final_project.git
    cd cmpe252_final_project/final_project
    ```
2.  **Create a Virtual Environment & Install:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    make install
    ```

## 🧪 Running Experiments

This repository uses **Hydra** for configuration management. All results, logs, and configurations are automatically saved to dynamically generated folders under `outputs/YYYY-MM-DD/HH-MM-SS/`.

You can use the provided `Makefile` for one-click execution:

```bash
# 1. Run the optimized PCA + Linear SVM pipeline
make train-svm

# 2. Train the CNN (ResNet-18) from scratch
make train-resnet

# 3. Fine-tune the Vision Transformer
make train-vit
```

*Want to test if the pipeline works without waiting hours? Run in debug mode:*
```bash
python src/main.py model=resnet18 debug_mode=true
```

## 📊 Results Summary
*(To be populated after final training runs)*

| Model | Test Accuracy | Training Time | Parameter Count |
| :--- | :--- | :--- | :--- |
| **PCA + Linear SVM** | ~XX% | ~XX mins | N/A |
| **ResNet-18 (Scratch)** | ~XX% | ~XX mins | ~11.1M |
| **ViT-B/16 (Fine-Tuned)**| ~XX% | ~XX mins | ~86M |

## 🏗 Architecture Decisions
*   **Configuration Management:** We chose **Hydra** over standard `argparse` because comparing three wildly different architectural approaches required distinct sets of hyperparameters. Hydra ensures reproducibility by saving the exact YAML state alongside the output artifacts.
*   **Engine Separation:** We abandoned traditional monolithic notebook scripts in favor of an idiomatic ML directory structure, separating PyTorch's epoch-based training loops from Scikit-Learn's procedural fit/predict flow.
*   **Algorithmic Optimization:** An RBF Kernel SVM on 50,000 samples has a time complexity approaching $O(N^3)$, which is computationally prohibitive. We intentionally utilized a `LinearSVC` ($O(N)$) following PCA to ensure the classical baseline is executable in a reasonable timeframe without OOM crashes.