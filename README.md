# HSLNet: A Line-Level Structure-Aware Method for Smart Contract Vulnerability Detection

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Note:** This repository contains the official implementation of **"A Line-Level Structure-Aware Method for Smart Contract Vulnerability Detection"** (Submitted to *Chinese Journal of Computers* / 《计算机学报》). 
> 
> *For double-blind review purposes, author identities and affiliations have been anonymized.*

## 📖 Introduction
Smart contract vulnerability detection models often treat code as flat token sequences or coarse-grained graph nodes, ignoring the crucial strict execution order and cross-line dependencies (e.g., Reentrancy relies heavily on the physical order of `call` and `state update`). 

**HSLNet** (Hierarchical Structured Learning Network) is a novel dual-branch framework that:
1. **Line-level Structure Modeling:** Explicitly captures the physical execution order of code lines using a Line-level Transformer with semantic-structure decoupling, eliminating the need for heavy static analysis tools (like Slither) and graph constructions.
2. **Global Semantic Contrastive Learning:** Maximizes the decision boundary between structurally similar but semantically divergent (vulnerable vs. safe) samples in a low-resource setting (10% few-shot scenario).

## 🚀 Key Features
- **End-to-End & Lightweight Preprocessing:** Directly takes raw Solidity text as input. Fast inference (~44ms/sample) without AST/CFG extraction overhead.
- **Micro-Order Sensitive:** Achieves highly sensitive detection for sequence-dependent vulnerabilities (e.g., CEI pattern violations).
- **Robust in Few-Shot:** Achieves state-of-the-art performance even when trained on only 10% of the labeled data.

---

## 🛠️ Requirements & Installation

Our experiments were conducted on Ubuntu 20.04 with an Intel Core i9-13900K and a single NVIDIA RTX 3090 (24GB VRAM).

```bash
# Clone the repository
git clone [https://github.com/Anonymous-submission/HSLNet.git](https://github.com/Anonymous-submission/HSLNet.git)
cd HSLNet

# Create a virtual environment
conda create -n hslnet python=3.7
conda activate hslnet

# Install dependencies
pip install torch torchvision torchaudio --extra-index-url [https://download.pytorch.org/whl/cu113](https://download.pytorch.org/whl/cu113)
pip install transformers scikit-learn numpy pandas tqdm
