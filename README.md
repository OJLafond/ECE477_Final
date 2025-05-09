# Stress Detection Using Wearable Sensor Data and Deep Neural Networks

This repository contains the full implementation of the paper  
**“Designing a Compact Stress Detection System for Edge Devices Using Wearable Sensor Data and Deep Neural Networks.”**

The project presents a complete pipeline for detecting stress in real-time using physiological signals from wearable medical sensors, optimized for execution on resource-constrained edge devices (e.g., smartwatches).

---

## Overview

The system performs binary classification ("stressed" or "not stressed") using a deep neural network trained on both real and synthetic physiological data.  
To reduce computational cost and support edge deployment, the network is dynamically optimized using the SCANN framework, which explores three architectural strategies:

- **Scheme A**: Constructive growth  
- **Scheme B**: Prune-regrow alternation  
- **Scheme C**: Fixed-layer structured pruning *(best-performing)*

All components of the pipeline—including preprocessing, augmentation, dimensionality reduction, model training, and evaluation—were run in **Google Colab**.

---

## Key Features

- **Wearable Sensor Input**: EDA, HR, SpO₂, skin temperature, tri-axial acceleration  
- **Dimensionality Reduction**: PCA and Random Projection (RP)  
- **Synthetic Data Augmentation**: Kernel Density Estimation (KDE) and Gaussian Mixture Models (GMM)  
- **Architecture Search**: SCANN-based neuron and connection growth, pruning, and retraining  
- **Evaluation Metrics**: Accuracy, F1 Score, AUC, Sparsity, Compactness

---

## Reproducing Results

1. Open `stress_detection_pipeline.ipynb` in [Google Colab](https://colab.research.google.com)
2. Run all cells to:
   - Preprocess the data
   - Generate synthetic data
   - Train SDNN with SCANN Schemes A, B, and C
   - Fine-tune the models on real data
   - Visualize confusion matrices, ROC curves, and compactness metrics
