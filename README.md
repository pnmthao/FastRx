# FastRx: Exploring Fastformer and Knowledge Graph for Personalized Medication Recommendations

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![Pytorch Version](https://img.shields.io/badge/Pytorch-1.13.1-orange)

## Table of Contents

- [Overview](#overview)
- [Usage Example](#usage-example)
- [Code Structure](#code-structure)
- [Dependencies](#dependencies)
- [Experimental Results](#experimental-results)
- [Contact Information](#contact-information)

---

## Overview

Personalized medication recommendation aims to suggest a set of medications according to a patient's clinical conditions. Not only should the patient's diagnosis, procedure, and medication history be considered, but drug-drug interactions (DDIs) should also be taken into account to avoid adverse drug reactions. Although recent studies on medication recommendation have considered drug-drug interactions and patient history, personalized disease progression and prescription have not been explicitly modeled. In this work, we proposed **FastRx, a medication recommendation model based on Fastformer combined with one-dimensional Convolutional Neural Network model (1D-CNN)** to capture longitudinality in patient history and Graph Convolutional Networks (GCNs) to handle DDI and co-prescribed medications in the Electronic Health Records (EHRs). Our experiments on the MIMIC-III dataset demonstrated that the proposed FastRx outperformed the existing state-of-the-art models for medication recommendation.

## Usage Example

```bash
conda create -n FastRx python=3.9.7
conda activate FastRx
pip install -r requirements.txt

# For training
./train.sh

# For testing (reproduce the paper's result)
./test.sh

```

## Code Structure
The codebase is organized as follows:
```bash
FastRx/
├── data/
├── saved/
├── tested/
├── ablations.py
├── FastRx.py
├── layers.py
├── models.py
├── README.md
├── requirements.txt
├── results.txt
├── test.sh
├── train.sh
└── util.py
```

## Dependencies
The code relies on the following libraries:

```Python
numpy==1.22.4
einops==0.6.0
torch==1.13.1
dill==0.3.6
prettytable==3.7.0
scikit-learn==1.2.2
tensorboard==2.11.2
```

<!-- ## Citation
If you use this code in your research, please cite it as follows:
.... -->

## Experimental Results

| Model             |       DDI  ↓       |     Jaccard ↑     |        F1 ↑       |      PRAUC ↑      | Avg.# of medication |
|-------------------|:---------------:|:---------------:|:---------------:|:---------------:|:-------------------:|
| FastRx w/o D      | 0.0673 ± 0.0008 | 0.5115 ± 0.0041 | 0.6676 ± 0.0038 | 0.7647 ± 0.0037 |   22.1290 ± 0.0156  |
| FastRx w/o P      | 0.0682 ± 0.0008 | 0.5204 ± 0.0037 | 0.6766 ± 0.0032 | 0.7761 ± 0.0035 |   23.1795 ± 0.2054  |
| FastRx w/o GCN    | 0.0685 ± 0.0008 | 0.5405 ± 0.0041 | 0.6932 ± 0.0036 | 0.7881 ± 0.0037 |   23.1122 ± 0.1861  |
| FastRx w/o 1D-CNN | 0.0691 ± 0.0007 | 0.5411 ± 0.0034 | 0.6938 ± 0.0030 | 0.7877 ± 0.0032 |   22.8025 ± 0.2047  |
| FastRx            | **0.0669 ± 0.0007** | **0.5443 ± 0.0037** | **0.6963 ± 0.0032** | **0.7882 ± 0.0037** |   23.0349 ± 0.1909  |

## Contact Information
For inquiries, collaboration, or support, please contact me:
- Thao Phan: pnmthao2908.ee10@nycu.edu.tw or pnmthaoct@gmail.com

## References
- This code is cloned from [SafeDrug](https://github.com/ycq091044/SafeDrug) - C. Yang, C. Xiao, F. Ma, L. Glass, and J. Sun,
“Safedrug: Dual molecular graph encoders for safe drug
recommendations,” CoRR, vol. abs/2105.02711, 2021. [Online].
Available: https://arxiv.org/abs/2105.02711
