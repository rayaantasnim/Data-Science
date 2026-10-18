<!-- Author: Rayaan tasnim, 13 aged prodigy -->

<div align="center">

# ⚡ Advanced Data Science & Deep Learning Suite

An elite collection of single-file, production-grade Python algorithmic architectures implemented from fundamental principles. Designed for extreme computational efficiency, research transparency, and minimal external friction.

<p align="center">
  <a href="https://github.com/rayaantasnim/Data-Science/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="License" />
  </a>
  <a href="https://www.python.org/downloads/release/python-3100/">
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  </a>
  <a href="https://github.com/rayaantasnim/Data-Science/actions">
    <img src="https://img.shields.io/badge/Build-Passing-brightgreen.svg?style=for-the-badge&logo=githubactions&logoColor=white" alt="Build Status" />
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/Code%20Style-Black-000000.svg?style=for-the-badge&logo=python&logoColor=white" alt="Code Style" />
  </a>
</p>

</div>

<hr />

## 📌 Project Overview

This repository houses ten single-file, foundational implementations of modern machine learning, deep learning, and agentic workflows. Built without abstraction abstractions or bloated frameworks, every module exposes raw mathematical primitives—ranging from dynamic autograd backpropagation loops and custom KV-cached multi-head attention blocks to variational Bayesian inference engines and online isolation trees.

### Core Architectural Principles
- **Algorithmic Transparency:** All math is fully unrolled—leveraging vectorization ($NumPy$, $PyTorch$, $TensorFlow$) for real-time clarity.
- **Zero Framework Bloat:** Self-contained architectures in clean, decoupled single files for seamless integration and auditability.
- **Strictly Typed & Verified:** Type hints, inline vector shape assertions, and rigorous numerical stability controls out of the box.

<hr />

## 📊 Dashboard & System Status

| ID | Architecture / Module | Domain | Autograd / Math Core | Status |
| :---: | :--- | :--- | :--- | :---: |
| **01** | [Self-Rerouting Agentic RAG](#1-self-rerouting-agentic-rag-with-real-time-data-drift-correction) | LLM Agents / Systems | Cosine Similarity & Drift Metrics | `STABLE` |
| **02** | [Physics-Informed Neural Nets](#2-differentiable-physics-informed-neural-networks-pinns) | Scientific Machine Learning | Dynamic Hessian / Laplacian Tensors | `STABLE` |
| **03** | [Transformer Architecture + KV Cache](#3-custom-transformer-architecture-from-scratch-with-kv-caching) | Natural Language Processing | Custom Scaled Dot-Product Attention | `STABLE` |
| **04** | [Bayesian Neural Networks](#4-bayesian-neural-networks-bnns-with-variational-inference) | Probabilistic Machine Learning | Kullback–Leibler ($\text{KL}$) Divergence Loss | `STABLE` |
| **05** | [Multi-Agent Reinforcement Learning](#5-multi-agent-reinforcement-learning-marl-for-market-simulation) | Multi-Agent Systems | Independent PPO / Actor-Critic | `STABLE` |
| **06** | [Streaming Online Isolation Forests](#6-streaming-anomaly-detection-with-online-isolation-forests) | Time Series / Anomaly | Dynamic Subsampling & Depth Traversal | `STABLE` |
| **07** | [Contrastive Learning (SimCLR)](#7-contrastive-self-supervised-learning-simclr-for-unlabeled-data) | Computer Vision / SSL | NT-Xent (Normalized Temperature-scaled Cross Entropy) | `STABLE` |
| **08** | [LLM Synthetic Data Generator](#8-end-to-end-llm-guided-synthetic-data-generation-pipeline) | Data Engineering / LLMs | Structured Pydantic Output Parsing | `STABLE` |
| **09** | [Graph Neural Networks (GNN)](#9-graph-neural-networks-gnn-for-fraud-ring-detection) | Graph Analytics | Custom Message-Passing Aggregator | `STABLE` |
| **10** | [Deep Generative Infilling (VAE)](#10-deep-generative-infilling-via-variational-autoencoders-vaes) | Generative Models | Reparameterization Trick ($z = \mu + \sigma \odot \epsilon$) | `STABLE` |

<hr />

## 🚀 The Architecture Suite

<details>
<summary><b>01. Self-Rerouting Agentic RAG with Real-Time Data Drift Correction</b></summary>

<br />

### Architectural Overview
An autonomous Retrieval-Augmented Generation (RAG) system featuring an internal feedback control loop. If context relevance drops below a dynamic percentile threshold or semantic drift is detected, the workflow automatically re-routes queries to secondary vector spaces or triggers online index re-clustering.

### Primary Math & Components
- Custom cosine similarity matrix vectorization in pure NumPy.
- Exponentially Weighted Moving Average (EWMA) tracking for query-to-document drift detection.
- Dynamic fallback routing logic bypassing hallucinations.

### Execution Command
```bash
python 01_agentic_rag_drift_correction.py --query "Evaluate system stability under dynamic query loads"
```
</details>

<details>
<summary><b>02. Differentiable Physics-Informed Neural Networks (PINNs)</b></summary>

<br />

### Architectural Overview
A deep learning system encoding physical laws governed by partial differential equations (PDEs), such as the Navier-Stokes or Burgers' equations, directly into the network's loss function.

### Primary Math & Components
- Explicit automatic differentiation via `torch.autograd.grad` to compute higher-order space/time derivatives ($\frac{\partial u}{\partial t}$, $\frac{\partial^2 u}{\partial x^2}$).
- Multi-objective composite loss functions ($L_{total} = L_{data} + \lambda L_{PDE}$).
- Continuous coordinate domain sampling routines.

### Execution Command
```bash
python 02_differentiable_pinn.py --epochs 2000 --viscous-coefficient 0.01
```
</details>

<details>
<summary><b>03. Custom Transformer Architecture from Scratch with KV Caching</b></summary>

<br />

### Architectural Overview
A ground-up implementation of a decoder-only Transformer module. Features custom multi-head self-attention, rotary positional embeddings (RoPE), and an active Key-Value (KV) cache for fast auto-regressive generation.

### Primary Math & Components
- Scaled Dot-Product Attention: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
- Explicit KV cache tensor allocation and updating across sequential generation steps.
- Masked causal attention operations executed via torch tensor manipulation.

### Execution Command
```bash
python 03_transformer_kv_cache.py --prompt "The mathematical foundation of intelligence is" --max-tokens 128
```
</details>

<details>
<summary><b>04. Bayesian Neural Networks (BNNs) with Variational Inference</b></summary>

<br />

### Architectural Overview
A neural network architecture that models weight uncertainty by replacing deterministic weights with probability distributions. Uses Stochastic Variational Inference (SVI) to approximate the intractable true posterior distribution.

### Primary Math & Components
- Reparameterization Trick for weight sampling ($W = \mu + \sigma \odot \epsilon$).
- Evidence Lower Bound (ELBO) optimization maximizing likelihood while penalizing KL Divergence:
  $$\text{ELBO} = \mathbb{E}_{q(\theta)}[\log p(\mathcal{D}|\theta)] - D_{KL}(q(\theta) \parallel p(\theta))$$
- Epistemic vs. Aleatoric uncertainty quantification pipelines.

### Execution Command
```bash
python 04_bayesian_neural_network.py --samples 500 --lr 0.005
```
</details>

<details>
<summary><b>05. Multi-Agent Reinforcement Learning (MARL) for Market Simulation</b></summary>

<br />

### Architectural Overview
A competitive-cooperative multi-agent reinforcement learning environment simulating order-book mechanics, continuous double auctions, and strategic liquidity provision using Multi-Agent Proximal Policy Optimization (MAPPO).

### Primary Math & Components
- Generalized Advantage Estimation (GAE) computed per individual agent.
- Actor-Critic neural architectures with shared background representations.
- Custom reward formulation handling slippage, inventory risk, and bid-ask spread optimization.

### Execution Command
```bash
python 05_marl_market_simulation.py --agents 8 --episodes 1000
```
</details>

<details>
<summary><b>06. Streaming Anomaly Detection with Online Isolation Forests</b></summary>

<br />

### Architectural Overview
An online variant of the Isolation Forest algorithm engineered for streaming time-series data. Continuously updates tree structures using sliding memory windows and dynamic path-length depth adjustments without needing full offline retrains.

### Primary Math & Components
- Incremental tree updating using random hyper-plane partitioning.
- Dynamic anomaly score derivation based on expected path length $c(n)$:
  $$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$
- Real-time sliding window statistical profiling.

### Execution Command
```bash
python 06_streaming_isolation_forest.py --window-size 5000 --contamination 0.01
```
</details>

<details>
<summary><b>07. Contrastive Self-Supervised Learning (SimCLR) for Unlabeled Data</b></summary>

<br />

### Architectural Overview
A self-supervised visual representation learning pipeline. Generates meaningful, low-dimensional feature embeddings from unlabeled visual data by maximizing agreement between differently augmented views of the same image.

### Primary Math & Components
- NT-Xent Loss (Normalized Temperature-scaled Cross Entropy):
  $$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{I}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$$
- Custom image transformation and stochastic augmentation pipeline.
- ResNet-style feature extractor backbone attached to a non-linear projection head.

### Execution Command
```bash
python 07_simclr_contrastive_learning.py --batch-size 256 --temperature 0.07
```
</details>

<details>
<summary><b>08. End-to-End LLM-Guided Synthetic Data Generation Pipeline</b></summary>

<br />

### Architectural Overview
A programmatic framework for synthesizing high-variance, domain-specific tabular and textual datasets. Utilizes structured schema enforcement, edge-case probabilistic sampling, and automatic validation against real data distributions.

### Primary Math & Components
- Strict JSON Schema enforcing via Pydantic model validation.
- Kolmogorov-Smirnov (KS) testing to verify statistical fidelity between real and synthetic distributions.
- Differential privacy noise injection mechanisms ($e^{\epsilon}$-privacy guarantees).

### Execution Command
```bash
python 08_llm_synthetic_data_pipeline.py --samples 10000 --out-format parquet
```
</details>

<details>
<summary><b>09. Graph Neural Networks (GNN) for Fraud Ring Detection</b></summary>

<br />

### Architectural Overview
A custom Graph Convolutional Network (GCN) built to detect anomalous subgraphs, money-laundering loops, and coordinated fraud rings in complex transactional networks.

### Primary Math & Components
- Graph Laplacian normalization: $\tilde{A} = D^{-\frac{1}{2}} (A + I_N) D^{-\frac{1}{2}}$.
- Custom message passing and neighborhood aggregation layers implemented using sparse tensor operations.
- Node classification and link prediction optimization pipelines.

### Execution Command
```bash
python 09_gnn_fraud_detection.py --nodes 10000 --edges 50000
```
</details>

<details>
<summary><b>10. Deep Generative Infilling via Variational Autoencoders (VAEs)</b></summary>

<br />

### Architectural Overview
A deep generative model designed to reconstruct corrupted or missing spatial data. Uses a probabilistic encoder-decoder architecture to infer the latent distribution of missing regions and reconstruct smooth, high-fidelity completions.

### Primary Math & Components
- Gaussian latent space parameterization ($\mu_\phi(x), \log\sigma_\phi^2(x)$).
- Reparameterization Trick enabling end-to-end backpropagation ($z = \mu + \sigma \odot \epsilon, \, \epsilon \sim \mathcal{N}(0, I)$).
- Binary Cross-Entropy combined with KL Divergence loss minimization.

### Execution Command
```bash
python 10_vae_generative_infilling.py --latent-dim 32 --mask-ratio 0.3
```
</details>

<hr />

## 🛠 Local Run Configuration

### Prerequisites
- **Python:** Version 3.10 or higher.
- **Git:** Version Control system.

### 1. Repository Acquisition
Clone the target repository locally:
```bash
git clone https://github.com/rayaantasnim/Data-Science.git
cd Data-Science
```

### 2. Environment Setup
Construct a isolated Python virtual environment:
```bash
# Initialize virtual environment
python3 -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows (PowerShell)
# .\venv\Scripts\Activate.ps1
```

### 3. Dependency Installation
Install all foundational dependencies required across the 10 modules:
```bash
pip install --upgrade pip
pip install numpy scipy torch tensorflow openai pydantic scikit-learn pandas
```

> [!WARNING]
> **Security Policy:** Never commit secrets or API keys to repository history. Configure environment variables securely in your shell before executing modules dependent on external LLM services:
> ```bash
> export OPENAI_API_KEY="your_actual_api_key_here"
> ```

<hr />

<div align="center">

**[Data-Science Repository](https://github.com/rayaantasnim/Data-Science)** • Authored by **Rayaan tasnim**

</div>