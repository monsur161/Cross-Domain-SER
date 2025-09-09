# A Comparative Study of CNN and Transformer Models for Cross-Domain Speech Emotion Recognition

## Project Summary
This project conducts an end-to-end investigation into the problem of domain generalization in Speech Emotion Recognition (SER).

The research followed a systematic, multi-stage process:
1. A **"Specialist" model** using a ResNet18 CNN on spectrograms was first developed, achieving high in-domain accuracy but failing catastrophically when tested on a different domain, thereby quantifying the "domain gap."
2. A **"Generalist" CNN model** was then created by training on a mixed-domain dataset. This approach was progressively enhanced with advanced techniques, culminating in our champion **"Ultimate Generalist"** which used a **balanced dataset** and **SpecAugment**, demonstrating near-perfect accuracy on the clean domain and strong generalization to the challenging domain.
3. Finally, this highly-optimized CNN approach was benchmarked against a state-of-the-art **HuBERT Speech Transformer**. This final experiment yielded a surprising and insightful conclusion: the curriculum learning strategy used for the Transformer led to catastrophic forgetting, making our **robustly-trained CNN model the superior solution** for this specific cross-domain task.

## Key Results
This table summarizes the performance of the most significant models developed in this study, showcasing the journey from a brittle specialist to a robust generalist and the final comparison with a Speech Transformer.

| Model Stage | Architecture | Data Strategy | RAVDESS Accuracy | CREMA-D Accuracy | IEMOCAP Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Specialist (L2)** | ResNet18 | RAVDESS Only | 80.56% | 22.39% | (N/A) |
| **Champion CNN (L4.5)** | **ResNet18** | **Balanced (R+C)** | **99.73%** | **62.71%** | (N/A) |
| **Transformer (L4.9)**| HuBERT | Multi-Stage (R+C -> I) | 45.86% | 44.71% | **66.60%** |

*(R=RAVDESS, C=CREMA-D, I=IEMOCAP)*

The key takeaway is the performance of our **Champion CNN (L4.5)**, which nearly perfected the clean RAVDESS dataset while simultaneously increasing the accuracy on the challenging CREMA-D dataset by over 40 percentage points compared to the Specialist baseline.

## Repository Structure
This project is documented in a series of Jupyter notebooks, each corresponding to a key phase of the research.

* **[Phase 1: Specialist Model Baseline](./notebooks/01_Specialist_Model_Baseline.ipynb):** Establishes a baseline by training a ResNet18 model on the clean RAVDESS dataset.
* **[Phase 2: Domain Gap and Initial Solutions](./notebooks/02_Domain_Gap_and_Initial_Solutions.ipynb):** Creates a more robust Specialist, quantifies the performance drop on the CREMA-D dataset, and tests Knowledge Distillation.
* **[Phase 3: Specialist Model Bias Analysis](./notebooks/03_Specialist_Model_Bias_Analysis.ipynb):** A fairness audit of the Specialist Model, revealing different error patterns for male and female speakers.
* **[Phase 4.3: First Generalist Model](./notebooks/04_Generalist_Model_v1_Bridging_the_Gap.ipynb):** Solves the domain gap by training a ResNet18 model on a combined RAVDESS and CREMA-D dataset.
* **[Phase 4.5: The Ultimate Generalist Model](./notebooks/05_Ultimate_Generalist_Model_v2.ipynb):** Achieves peak performance by introducing data balancing, SpecAugment, and an advanced learning rate scheduler.
* **[Phase 4.6: Unbalanced Data Control Experiment](./notebooks/06_Generalist_Ablation_Unbalanced_Data.ipynb):** An ablation study proving the critical importance of the data balancing strategy.
* **[Phase 4.7: EfficientNet Control Experiment](./notebooks/07_Generalist_Ablation_EfficientNet.ipynb):** A final test to confirm the performance ceiling of the CNN approach, showing that a more advanced architecture does not yield significant gains.
* **[Phase 4.8: The Transformer Paradigm Shift (Failed Attempt)](./notebooks/08_Transformer_v1_Wav2Vec2_Failed_Attempt.ipynb):** Documents the first attempt to use an end-to-end Speech Transformer, which failed due to training instability (`NaN` loss).
* **[Phase 4.9: The Final HuBERT Experiment](./notebooks/09_Transformer_v2_HuBERT_Final_Experiment.ipynb):** Successfully implements and trains the HuBERT model, providing the final, conclusive results and comparison for the project.

## Setup and Usage
This repository contains the Jupyter notebooks used to conduct the research. To reproduce the results, please follow these steps.

### 1. Prerequisites
* A system with a CUDA-enabled GPU is highly recommended for training the models in a reasonable amount of time.
* Python 3.10+

### 2. Clone the Repository
```bash
git clone https://github.com/monsur161/Cross-Domain-SER.git
cd Cross-Domain-SER
