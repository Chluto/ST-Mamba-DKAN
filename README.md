# ST-Mamba-DKAN: Multi-Modal Spatial-Temporal Traffic Flow Prediction

> ⚠️ **Note:** This paper has not been published yet. To protect the intellectual property and the integrity of the blind review process, this repository provides the core algorithm implementation (ST-Mamba-DKAN), the **real-world multi-modal dataset** (Customized PeMS + Meteorology), and the raw data preprocessing pipeline for transparency and conceptual verification.
> 
> For the complete, highly-optimized, and fully reproducible project codebase configured for Linux server environments (including high-performance CUDA extensions, early-stopping mechanisms, and optimal hyperparameter logs), please contact the author directly via email at: **jiechao_aca@163.com**

> ⚠️ **声明：** 本论文目前还未发表。为了保护知识产权及盲审过程的公正性，本仓库仅开源核心算法的极简实现（ST-Mamba-DKAN）、**真实多模态数据集**（自定义 PeMS + 气象）以及原始数据预处理管线，以供算法透明度展示与概念验证。
> 
> 如需获取配置于 Linux 服务器环境的完整、高度优化且完全可复现的项目源码（包含高性能 CUDA 算子扩展、早停机制及最优超参数日志），请直接通过邮件联系作者：**jiechao_aca@163.com**

---

## 📖 Introduction (简介)

This repository contains the minimalist, unoptimized PyTorch implementation of **ST-Mamba-DKAN**, a novel multi-modal framework designed for long-term traffic flow prediction under complex meteorological conditions. By seamlessly integrating the State Space Model (Mamba) for linear-complexity temporal sequence modeling and the Discrete Kolmogorov-Arnold Network (DKAN) for non-linear spatial topology aggregation, along with a Decoupled Gated Late Fusion (DGLF) mechanism, our model achieves State-of-the-Art performance while effectively preventing "overshoot" errors during extreme weather events.

本项目提供了 **ST-Mamba-DKAN** 的极简（未深度优化）PyTorch 实现版本。这是一种新型的多模态时空预测框架，专为复杂气象条件下的长程交通流预测而设计。通过将具备线性复杂度的状态空间模型（Mamba）、擅长捕捉非线性拓扑的离散 Kolmogorov-Arnold 网络（DKAN）以及解耦门控后期融合机制（DGLF）进行无缝集成，本架构不仅实现了多步长预测的 SOTA 性能，更有效克服了极端恶劣天气引发的“流量过冲（Overshoot）”误差。

---

## 📂 Repository Structure (项目结构)

```text
github_code/
├── dataset/                                 # Real-world Multi-modal Dataset
│   ├── pemsd4_subset.npz                    # Custom PeMS spatial-temporal physical signals subset
│   ├── weather_raw.csv                      # Synchronized historical meteorological data (e.g., rainfall, temp)
│   └── aligned_multimodal_features.npy      # Cached tensor after data alignment and preprocessing
├── model.py                                 # Core definitions of ST-Mamba-DKAN, DKAN Layer, and DGLF
├── data_preprocess.py                       # Data pipeline for timestamp alignment and normalization
├── data_loader.py                           # Dataset mounting classes and multimodal data iterators
├── simple_train.py                          # Out-of-the-box minimal training loop for conceptual verification
└── requirements.txt                         # Minimal environment dependencies