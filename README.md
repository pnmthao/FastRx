# FastRx: Exploring Fastformer and Knowledge Graph for Personalized Medication Recommendations

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![Pytorch Version](https://img.shields.io/badge/Pytorch-1.13.1-orange)

## Table of Contents

- [FastRx: Exploring Fastformer and Knowledge Graph for Personalized Medication Recommendations](#fastrx-exploring-fastformer-and-knowledge-graph-for-personalized-medication-recommendations)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Usage Example](#usage-example)
  - [Code Structure](#code-structure)
  - [Dependencies](#dependencies)
  - [Experimental Results](#experimental-results)
    - [Performance Comparison of Different Methods on MIMIC-III dataset](#performance-comparison-of-different-methods-on-mimic-iii-dataset)
    - [Ablation Study for FastRx on MIMIC-III dataset](#ablation-study-for-fastrx-on-mimic-iii-dataset)
  - [Contact Information](#contact-information)
  - [References](#references)

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

### Performance Comparison of Different Methods on MIMIC-III dataset
| Method   | DDI ↓          | Jaccard ↑        | F1 ↑             | PRAUC ↑          | Avg.# of Drugs       |
|----------|:----------------:|:------------------:|:------------------:|:------------------:|:----------------------:|
| LR       | 0.0829 ± 0.0009| 0.4865 ± 0.0021  | 0.6434 ± 0.0019  | 0.7509 ± 0.0018  | 16.1773 ± 0.0942     |
| ECC [^1]     | 0.0846 ± 0.0018| 0.4996 ± 0.0049  | 0.6569 ± 0.0044  | 0.6844 ± 0.0038  | 18.0722 ± 0.1914     |
| LEAP [^2]    | 0.0731 ± 0.0008| 0.4521 ± 0.0024  | 0.6138 ± 0.0026  | 0.6549 ± 0.0033  | 18.7138 ± 0.0666     |
| DMNC [^3]    | 0.0842 ± 0.0011| 0.4864 ± 0.0025  | 0.6529 ± 0.0030  | 0.7580 ± 0.0039  | 20.0000 ± 0.0000     |
| RETAIN [^4]  | 0.0835 ± 0.0020| 0.4887 ± 0.0028  | 0.6481 ± 0.0027  | 0.7556 ± 0.0033  | 20.4051 ± 0.2832     |
| GAMENet [^5]  | 0.0864 ± 0.0006| 0.5067 ± 0.0025  | 0.6626 ± 0.0025  | 0.7631 ± 0.0030  | 27.2145 ± 0.1141     |
| 4SDrug [^6]   | 0.0703 ± 0.0011| 0.4800 ± 0.0027  | 0.6404 ± 0.0024  | 0.7611 ± 0.0026  | 16.1684 ± 0.1280     |
| MICRON [^7]   | 0.0641 ± 0.0007| 0.5100 ± 0.0033  | 0.6654 ± 0.0031  | 0.7631 ± 0.0026  | 17.9267 ± 0.2172     |
| SafeDrug [^8] | **0.0589 ± 0.0005**| 0.5213 ± 0.0030  | 0.6768 ± 0.0027  | 0.7647 ± 0.0025  | 19.9178 ± 0.1604     |
| DrugRec [^9] | *0.0597 ± 0.0006*| 0.5220 ± 0.0034  | 0.6771 ± 0.0031  | 0.7720 ± 0.0036  | 22.0006 ± 0.1604     |
| MoleRec [^10] | 0.0756 ± 0.0006| 0.5301 ± 0.0025  | 0.6841 ± 0.0022  | 0.7748 ± 0.0022  | 22.2239 ± 0.1661     |
| COGNet [^11]  | 0.0858 ± 0.0008| 0.5316 ± 0.0020  | 0.6644 ± 0.0018  | 0.7707 ± 0.0021  | 27.6279 ± 0.0802     |
| StratMed [^12] | 0.0642 ± 0.0005| *0.5321 ± 0.0035*  | *0.6861 ± 0.0034*  | *0.7779 ± 0.0043*  | 20.5318 ± 0.1681     |
| FastRx   | 0.0669 ± 0.0007| **0.5443 ± 0.0037**  | **0.6963 ± 0.0032**  | **0.7882 ± 0.0037**  | 23.0349 ± 0.1909     |


### Ablation Study for FastRx on MIMIC-III dataset
| Model             |       DDI  ↓       |     Jaccard ↑     |        F1 ↑       |      PRAUC ↑      | Avg.# of Drugs |
|-------------------|:---------------:|:---------------:|:---------------:|:---------------:|:-------------------:|
| FastRx w/o $\mathcal{D}$      | 0.0673 ± 0.0008 | 0.5115 ± 0.0041 | 0.6676 ± 0.0038 | 0.7647 ± 0.0037 |   22.1290 ± 0.0156  |
| FastRx w/o $\mathcal{P}$      | 0.0682 ± 0.0008 | 0.5204 ± 0.0037 | 0.6766 ± 0.0032 | 0.7761 ± 0.0035 |   23.1795 ± 0.2054  |
| FastRx w/o GCN    | 0.0685 ± 0.0008 | 0.5405 ± 0.0041 | 0.6932 ± 0.0036 | *0.7881 ± 0.0037* |   23.1122 ± 0.1861  |
| FastRx w/o 1D-CNN | 0.0691 ± 0.0007 | *0.5411 ± 0.0034* | *0.6938 ± 0.0030* | 0.7877 ± 0.0032 |   22.8025 ± 0.2047  |
| FastRx w/o RNN | **0.0585 ± 0.0005** | 0.5250 ± 0.0044 | 0.6798 ± 0.0039 | 0.7761 ± 0.0038 | 21.6724 ± 0.2034 |
| FastRx w/o Transformer | 0.0685 ± 0.0008 | 0.5294 ± 0.0051 | 0.6831 ± 0.0045 | 0.7802 ± 0.0038 | 23.6528 ± 0.2228  |
| FastRx            | *0.0669 ± 0.0007* | **0.5443 ± 0.0037** | **0.6963 ± 0.0032** | **0.7882 ± 0.0037** |   23.0349 ± 0.1909  |


## Contact Information
For inquiries, collaboration, or support, please contact me:
- Thao Phan: pnmthao2908.ee10@nycu.edu.tw or pnmthaoct@gmail.com

## References
> [!NOTE]
> This code is cloned from [GAMENet](https://github.com/sjy1203/GAMENet) - Shang, J., Xiao, C., Ma, T., Li, H., & Sun, J. (2019, July). Gamenet: Graph augmented memory networks for recommending medication combination. In *proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 33, No. 01, pp. 1126-1133).


[^1]: Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier chains for multi-label classification. Machine learning, 85, 333-359.

[^2]: Zhang, Y., Chen, R., Tang, J., Stewart, W. F., & Sun, J. (2017, August). LEAP: learning to prescribe effective and safe treatment combinations for multimorbidity. In proceedings of the 23rd ACM SIGKDD international conference on knowledge Discovery and data Mining (pp. 1315-1324).

[^3]: Le, H., Tran, T., & Venkatesh, S. (2018, July). Dual memory neural computer for asynchronous two-view sequential learning. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 1637-1645).

[^4]: Choi, E., Bahadori, M. T., Sun, J., Kulas, J., Schuetz, A., & Stewart, W. (2016). Retain: An interpretable predictive model for healthcare using reverse time attention mechanism. Advances in neural information processing systems, 29.

[^5]: Shang, J., Xiao, C., Ma, T., Li, H., & Sun, J. (2019, July). Gamenet: Graph augmented memory networks for recommending medication combination. In proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 1126-1133).

[^6]: Tan, Y., Kong, C., Yu, L., Li, P., Chen, C., Zheng, X., ... & Yang, C. (2022, August). 4sdrug: Symptom-based set-to-set small and safe drug recommendation. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 3970-3980).

[^7]: Yang, C., Xiao, C., Glass, L., & Sun, J. (2021). Change matters: Medication change prediction with recurrent residual networks. arXiv preprint arXiv:2105.01876.

[^8]: Yang, C., Xiao, C., Ma, F., Glass, L., & Sun, J. (2021). Safedrug: Dual molecular graph encoders for recommending effective and safe drug combinations. arXiv preprint arXiv:2105.02711.

[^9]: Sun, H., Xie, S., Li, S., Chen, Y., Wen, J. R., & Yan, R. (2022). Debiased, longitudinal and coordinated drug recommendation through multi-visit clinic records. Advances in Neural Information Processing Systems, 35, 27837-27849.

[^10]: Yang, N., Zeng, K., Wu, Q., & Yan, J. (2023, April). Molerec: Combinatorial drug recommendation with substructure-aware molecular representation learning. In Proceedings of the ACM Web Conference 2023 (pp. 4075-4085).

[^11]: Wu, R., Qiu, Z., Jiang, J., Qi, G., & Wu, X. (2022, April). Conditional generation net for medication recommendation. In Proceedings of the ACM Web Conference 2022 (pp. 935-945).

[^12]: Li, X., Liang, S., Hou, Y., & Ma, T. (2024). StratMed: Relevance stratification between biomedical entities for sparsity on medication recommendation. Knowledge-Based Systems, 284, 111239.
