# Deep Reinforcement Learning for Cost-Effective Medical Diagnosis

This repository implements an end-to-end pipeline for cost-effective medical diagnosis using deep reinforcement learning (RL) combined with advanced imputation techniques. The project is inspired by the paper:

> **Deep Reinforcement Learning for Cost-Effective Medical Diagnosis**  
> Zheng Yu, Yikuan Li, Joseph Kim, Kaixuan Huang, Yuan Luo, Mengdi Wang  
> [arXiv:2302.10261](https://arxiv.org/abs/2302.10261)

In our implementation, we focus on diagnosing diabetes using the NHANES dataset. Our system automatically decides which lab tests to order, balancing the cost of tests with diagnostic accuracy.

---

## Table of Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Baseline Models](#baseline-models)
- [Imputation Module](#imputation-module)
- [Environment](#environment)
- [Training](#training) 
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

---

## Overview

The primary goal of this project is to develop a system that minimizes false negatives while reducing the cost of medical testing. The system is composed of several components:

- **Data Merging & Preprocessing:**  
  Merge multiple NHANES datasets and preprocess features (including normalization and one-hot encoding of categorical data).

- **Baselines:**  
  Train classical machine learning models (Random Forest, Logistic Regression, XGBoost) for diabetes diagnosis as a baseline.

- **Imputation Module:**  
  Use a Normalizing Flow–based imputation model to fill in missing lab test values.

- **Reinforcement Learning Environment:**  
  A custom Gym environment simulates the sequential decision process of ordering test panels (each with an associated cost) and making a diagnosis. The reward function incorporates:
  - **Dynamic cost scaling:** Ordering later tests costs more.
  - **Adaptive early diagnosis penalty:** Diagnosing with too few tests ordered incurs a penalty.
  - **Bonus for additional tests:** Ordering extra tests beyond a minimum threshold gives a bonus.

- **RL Agent Training:**  
  Train an RL agent using Maskable PPO to learn an optimal policy that minimizes false negatives and overall cost.

---

## Data Preparation
- **Merge data:**
The merge.py script reads several NHANES XPT files (demographics, lab tests, etc.), selects relevant columns, handles missing values, and merges them into a single CSV file (nhanes_diabetes.csv).
To run:
> python merge.py

---

- **Preprocessing:**
The preprocessing.py script loads the merged data, applies one-hot encoding (e.g., for Gender), normalizes medical test values (excluding age), and saves the final dataset (nhanes_preprocessed.csv) for model training.
To run:
> python preprocessing.py

---

## Baseline Models
The baselines.py script implements baseline classifiers (Random Forest, Logistic Regression, XGBoost) on the preprocessed data. It performs training, cross-validation, and plots feature importances and ROC curves.
To run:
> python baselines.py

---

## Imputation Module
The imputation module is implemented in imputation.py, n_flow.py, and flow_models.py.
- **n_flow.py:** Contains normalizing flow blocks (split, merge, permutation, and affine coupling layers).
- **flow_models.py:** Defines the MLP used as the parameter network in the affine coupling layers.
- **imputation.py:** Implements the imputer class that trains the normalizing flow to impute missing lab test values.
You can test and fine-tune the imputation module by running the test code (commented out at the bottom of imputation.py).

---

## Environment
The custom environment is defined in environment.py. It simulates the sequential decision process:
- **Actions:** Order a test panel or make a diagnosis.
- **Observations:** Concatenated imputed test values and a diagnosis flag.
- **Reward Function:** Combines dynamic cost scaling, an adaptive penalty for diagnosing too early, and a bonus for ordering extra tests.

---

## Training
The RL training is conducted via train.py:
- The agent is trained using Maskable PPO (from sb3-contrib).
- Diagnosis rewards are set by λ = 5 or λ = 10 for a correct positive diagnosis and test cost penalties are scaled by ρ = 0.1 or ρ = 0.2.
- The training loop trains the imputer, classifier, and RL agent sequentially.
To start training:
> python train.py

---

## Evaluation
The evaluate.py script loads trained RL models and evaluates them over many episodes. It computes key metrics such as:
- Accuracy, F1 score, false negatives,
- Average number of panels ordered,
- Average cost per episode.
It also generates plots (Pareto front and bar charts) to compare different hyperparameter configurations.
To run evaluation:
> python evaluate.py

---

## Usage

### Requirements

- Python 3.8 or later
- [PyTorch](https://pytorch.org/) (CPU/GPU)
- numpy, pandas, matplotlib, seaborn, tqdm
- scikit-learn
- [gym](https://www.gymlibrary.ml/) (or Gymnasium)
- [stable-baselines3](https://stable-baselines3.readthedocs.io/) and [sb3-contrib](https://sb3-contrib.readthedocs.io/)
- [pyreadstat](https://github.com/Roche/pyreadstat) for reading NHANES XPT files
- tensorboardX

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ComprisedAxis/Leveraging-Reinforcement-Learning-for-Cost-Effective-Medical-Diagnostics.git
   cd Leveraging-Reinforcement-Learning-for-Cost-Effective-Medical-Diagnostics
   pip install -r requirements.txt
2. Place the NHANES raw data files (e.g., P_DEMO.xpt, P_DIQ.xpt, etc.) in the data/ folder.

---

## Results
In our experiments, we observed that:
- Setting a higher true-positive reward significantly improved the F1 score and reduced false negatives.
- A moderate cost penalty scaling provided the best trade-off between ordering sufficient tests and minimizing unnecessary costs.
- The RL agent learns to order test panels dynamically based on the cost–accuracy trade-off, outperforming baseline static classifiers in terms of reducing false negatives.

---

## References
> Zheng Yu, Yikuan Li, Joseph Kim, Kaixuan Huang, Yuan Luo, Mengdi Wang.
> Deep Reinforcement Learning for Cost-Effective Medical Diagnosis
> arXiv:2302.10261
- Stable-Baselines3 Documentation
- Gym Library
- PyReadStat Documentation
- Additional references as cited in the paper.
