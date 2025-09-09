# A Comparative Study of CNN and Transformer Models for Cross-Domain Speech Emotion Recognition

## Project Summary
This project conducts an end-to-end investigation into the problem of domain generalization in Speech Emotion Recognition (SER). We begin by establishing a baseline "Specialist" model using a CNN on the RAVDESS dataset, demonstrating its high performance in-domain and its failure to generalize to the CREMA-D dataset. We then develop a series of "Generalist" models, culminating in a highly-optimized CNN that successfully bridges the domain gap. Finally, we compare our champion CNN against a state-of-the-art HuBERT Speech Transformer, leading to a surprising conclusion about the effectiveness of different architectural paradigms for this task.

## Key Results
This table summarizes the performance of the most significant models developed in this study.

| Model | Architecture | Data Strategy | RAVDESS Accuracy | CREMA-D Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Specialist (L2)** | ResNet18 | RAVDESS Only | 80.56% | 22.39% |
| **Champion CNN (L4.5)**| ResNet18 | **Balanced** | **99.73%** | **62.71%** |
| **Transformer (L4.9)**| HuBERT | Multi-Stage | 45.86% | 44.71% |

## Repository Structure
This project is documented in a series of Jupyter notebooks, each corresponding to a key phase of the research.

* **[Phase 1: Specialist Model Baseline](./notebooks/01_Specialist_Model_Baseline.ipynb):** Establishes a baseline by training a ResNet18 model on the clean RAVDESS dataset.
* **[Phase 2: Domain Gap and Initial Solutions](./notebooks/02_Domain_Gap_and_Initial_Solutions.ipynb):** Creates a more robust Specialist, quantifies the performance drop on the CREMA-D dataset, and tests Knowledge Distillation.
* **[Phase 3: Specialist Model Bias Analysis](./notebooks/03_Specialist_Model_Bias_Analysis.ipynb):** A fairness audit of the Specialist Model, revealing different error patterns for male and female speakers.
* **[Phase 4.3: First Generalist Model](./notebooks/04_Generalist_Model_v1_Bridging_the_Gap.ipynb):** Solves the domain gap by training a ResNet18 model on a combined RAVDESS and CREMA-D dataset.
* **[Phase 4.5: The Ultimate Generalist Model](./notebooks/05_Ultimate_Generalist_Model_v2.ipynb):** Achieves peak performance by introducing data balancing, SpecAugment, and an advanced learning rate scheduler.
* **[Phase 4.6: Unbalanced Data Control Experiment](./notebooks/06_Generalist_Ablation_Unbalanced_Data.ipynb):** An ablation study proving the critical importance of the data balancing strategy.
* **[Phase 4.7: EfficientNet Control Experiment](./notebooks/07_Generalist_Ablation_EfficientNet.ipynb):** A final test to confirm the performance ceiling of the CNN approach, showing that a more advanced architecture does not yield significant gains.

## Setup and Usage
[Content to be added here: Instructions on how to set up the environment using requirements.txt and run the code.]
