# BaEF: Building-aware Global ITransformer Framework for Multi-Building Energy Forecasting

本仓库是面向论文主线整理后的 BaEF 代码仓库：

**从传统 Building Energy Forecasting（BEF）进一步明确提出 Building-aware Energy Forecasting（BaEF）问题设定，并在 BDG2 和 BDG1 的 250 建筑协议上，使用 GlobalITransformer-building-aware framework 系统检验建筑元数据在 seen-building 与 unseen-building 场景中的作用边界。**

This repository contains the cleaned research codebase for our BaEF paper line:

**From traditional Building Energy Forecasting (BEF) to Building-aware Energy Forecasting (BaEF), with a 250-building benchmark on BDG2 and BDG1, and a GlobalITransformer-building-aware framework for studying the boundary of building metadata under seen-building and unseen-building settings.**

## 1. 电脑配置 / Hardware and Software

### 中文

以下为本项目开发与实验整理时使用的主要环境，可作为复现实验的参考配置：

| 项目 | 配置 |
| --- | --- |
| 操作系统 | Windows 10 64-bit |
| CPU | Intel Core i7-12800HX |
| CPU 核心 / 线程 | 16 Cores / 24 Threads |
| 内存 | 32 GB RAM |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Python | 3.10.20 |
| PyTorch | 2.11.0 + CUDA 12.8 |
| NumPy | 2.2.6 |
| Pandas | 2.3.3 |
| Scikit-learn | 1.7.2 |
| LightGBM | 4.6.0 |

说明：

- 本项目优先支持 GPU 训练，但也可以在 CPU 上运行。
- 对于 LSTM、PatchTST、iTransformer 及其 building-aware 版本，推荐使用 NVIDIA GPU。
- LightGBM 与数据预处理步骤对 GPU 没有硬性要求。

### English

The following machine was used during code cleanup and experiment organization. It can be treated as a reference environment for reproduction:

| Item | Specification |
| --- | --- |
| OS | Windows 10 64-bit |
| CPU | Intel Core i7-12800HX |
| CPU Cores / Threads | 16 Cores / 24 Threads |
| Memory | 32 GB RAM |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Python | 3.10.20 |
| PyTorch | 2.11.0 + CUDA 12.8 |
| NumPy | 2.2.6 |
| Pandas | 2.3.3 |
| Scikit-learn | 1.7.2 |
| LightGBM | 4.6.0 |

Notes:

- GPU training is recommended but not mandatory.
- LSTM, PatchTST, iTransformer, and building-aware iTransformer benefit substantially from an NVIDIA GPU.
- LightGBM and data preprocessing can run on CPU only.

## 2. 代码框架 / Code Structure

### 中文

当前 GitHub 版本只保留与论文主线直接相关的最小可复现代码，不再包含历史探索路线、缓存和训练产物。

```text
Building-aware Energy Forecasting/
├── Building Data Genome Project 1/          # 本地数据目录，默认不上传 GitHub
├── Building Data Genome Project 2/          # 本地数据目录，默认不上传 GitHub
├── configs/                                # 论文主线实验配置
│   ├── global_lightgbm_250.json
│   ├── global_lstm_250.json
│   ├── global_patchtst_250.json
│   ├── global_itransformer_250.json
│   ├── building_aware_itransformer_250_noweather.json
│   ├── global_lightgbm_bdg1_250.json
│   ├── global_lstm_bdg1_250.json
│   ├── global_patchtst_bdg1_250.json
│   ├── global_itransformer_bdg1_250.json
│   ├── building_aware_itransformer_bdg1_250_noweather.json
│   ├── building_aware_unseen_itransformer_250_noweather.json
│   ├── building_aware_unseen_itransformer_bdg1_250_noweather.json
│   ├── paper_benchmark_250_manifest.json
│   └── paper_benchmark_bdg1_250_manifest.json
├── scripts/                                # 训练、评估与汇总入口
│   ├── prepare_bdg1_2015_canonical.py
│   ├── run_global_lightgbm.py
│   ├── run_global_lstm.py
│   ├── run_global_patchtst.py
│   ├── run_global_itransformer.py
│   ├── run_building_aware_itransformer.py
│   ├── run_multiseed_building_aware.py
│   ├── run_unseen_building_itransformer.py
│   ├── run_multisplit_unseen_building.py
│   ├── summarize_paper_benchmark.py
│   ├── summarize_multiseed_building_aware.py
│   ├── summarize_multisplit_unseen_building.py
│   └── summarize_by_building_type.py
├── src/building_aware/                     # 核心模型与数据处理模块
│   ├── baselines.py
│   ├── data_utils.py
│   ├── features.py
│   ├── global_lightgbm.py
│   ├── itransformer.py
│   ├── lstm.py
│   ├── metrics.py
│   ├── patchtst.py
│   └── time_splits.py
├── outputs/                                # 运行时输出目录，默认不上传训练结果
├── .gitignore
├── README.md
└── requirements.txt
```

各模块功能如下：

- `configs/`：论文主线的实验协议配置。
- `scripts/`：所有训练、评估、汇总的命令入口。
- `src/building_aware/`：数据处理、特征工程、基线模型、iTransformer、PatchTST、评估指标等核心实现。
- `outputs/`：模型检查点、结果表、预测样本、图表等运行产物。

### English

The current GitHub version keeps only the paper-critical and minimally reproducible workflow. Historical exploratory branches, cache folders, and generated training artifacts have been removed.


Core directories:

- `configs/`: experiment configurations used in the paper workflow.
- `scripts/`: training, evaluation, and result summarization entry points.
- `src/building_aware/`: core implementations for data loading, feature engineering, baselines, iTransformer, PatchTST, metrics, and time splitting.
- `outputs/`: runtime outputs such as checkpoints, predictions, logs, and result tables.

## 3. 数据集下载 / Dataset Download

### 中文

本项目不直接提供原始数据。请先下载 BDG2 和 BDG1，并按如下方式放到仓库根目录：

```text
Building-aware Energy Forecasting/
├── Building Data Genome Project 1/
└── Building Data Genome Project 2/
```

推荐下载来源：

- **BDG2 official GitHub**: <https://github.com/buds-lab/building-data-genome-project-2>
- **BDG1 official GitHub**: <https://github.com/buds-lab/the-building-data-genome-project>

（或者在kaggle官网下载）

数据说明：

- **BDG2**：主实验数据集，用于 250 建筑的 seen-building 主表和 unseen-building 边界检验。
- **BDG1**：跨数据集验证数据集，用于检验 BaEF 设定在另一开放楼宇数据集上的稳定性与边界。

对于 BDG1，本仓库默认使用整理后的 2015 canonical 文件。如果你下载的是原始仓库版本，需要先执行：

```powershell
python .\scripts\prepare_bdg1_2015_canonical.py --data-dir ".\Building Data Genome Project 1"
```

### English

Raw datasets are not shipped with this repository. Please download BDG2 and BDG1 and place them under the repository root as:

```text
Building-aware Energy Forecasting/
├── Building Data Genome Project 1/
└── Building Data Genome Project 2/
```

Recommended sources:

- **BDG2 official GitHub**: <https://github.com/buds-lab/building-data-genome-project-2>
- **BDG1 official GitHub**: <https://github.com/buds-lab/the-building-data-genome-project>

（or download on kaggle）

Dataset roles in this project:

- **BDG2** is the main benchmark for the 250-building seen-building experiments and unseen-building boundary tests.
- **BDG1** is used as a cross-dataset validation benchmark for BaEF.

For BDG1, this repository expects canonical 2015 files. If you start from the raw BDG1 repository, run:

```powershell
python .\scripts\prepare_bdg1_2015_canonical.py --data-dir ".\Building Data Genome Project 1"
```

## 4. 环境安装 / Environment Setup

### 中文

推荐使用 Anaconda 或 Miniconda。

#### 4.1 创建环境

```powershell
conda create -n baef python=3.10 -y
conda activate baef
```

#### 4.2 安装 PyTorch

如果你使用 NVIDIA GPU，建议先根据你的 CUDA 版本从 PyTorch 官方页面安装对应版本的 PyTorch。

当前开发环境参考版本为：

- PyTorch `2.11.0`
- CUDA `12.8`

安装完成后，可检查：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

#### 4.3 安装其余依赖

```powershell
pip install -r requirements.txt
```

### English

We recommend using Anaconda or Miniconda.

#### 4.1 Create the environment

```powershell
conda create -n baef python=3.10 -y
conda activate baef
```

#### 4.2 Install PyTorch

If you use an NVIDIA GPU, install the PyTorch build that matches your CUDA version from the official PyTorch installation page.

Reference environment used during development:

- PyTorch `2.11.0`
- CUDA `12.8`

Sanity check:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

#### 4.3 Install the remaining dependencies

```powershell
pip install -r requirements.txt
```

## 5. 训练与评估 / Training and Evaluation

### 中文

先进入项目目录：

```powershell
cd "D:\DL\BEF\Building-aware Energy Forecasting"（请修改为自己的目录）
```

### 5.1 BDG2：250 建筑 seen-building 主实验

运行六个主方法中的五个训练脚本，然后汇总主表：

```powershell
python .\scripts\run_global_lightgbm.py
python .\scripts\run_global_lstm.py
python .\scripts\run_global_patchtst.py
python .\scripts\run_global_itransformer.py
python .\scripts\run_building_aware_itransformer.py
python .\scripts\summarize_paper_benchmark.py
```

### 5.2 BDG1：250 建筑 seen-building 跨数据集验证

如果 BDG1 还未转换为 canonical 2015 格式，先执行：

```powershell
python .\scripts\prepare_bdg1_2015_canonical.py --data-dir ".\Building Data Genome Project 1"
```

然后运行：

```powershell
python .\scripts\run_global_lightgbm.py --config .\configs\global_lightgbm_bdg1_250.json
python .\scripts\run_global_lstm.py --config .\configs\global_lstm_bdg1_250.json
python .\scripts\run_global_patchtst.py --config .\configs\global_patchtst_bdg1_250.json
python .\scripts\run_global_itransformer.py --config .\configs\global_itransformer_bdg1_250.json
python .\scripts\run_building_aware_itransformer.py --config .\configs\building_aware_itransformer_bdg1_250_noweather.json
python .\scripts\summarize_paper_benchmark.py --manifest .\configs\paper_benchmark_bdg1_250_manifest.json
```

### 5.3 Multi-seed 稳定性检验

BDG2：

```powershell
python .\scripts\run_multiseed_building_aware.py
python .\scripts\summarize_multiseed_building_aware.py
```

BDG1：

```powershell
python .\scripts\run_multiseed_building_aware.py --base-config .\configs\building_aware_itransformer_bdg1_250_noweather.json
python .\scripts\summarize_multiseed_building_aware.py --base-run-name building_aware_itransformer_bdg1_250_noweather
```

### 5.4 Unseen-building 边界检验

BDG2：

```powershell
python .\scripts\run_unseen_building_itransformer.py
python .\scripts\run_multisplit_unseen_building.py
python .\scripts\summarize_multisplit_unseen_building.py
```

BDG1：

```powershell
python .\scripts\run_unseen_building_itransformer.py --config .\configs\building_aware_unseen_itransformer_bdg1_250_noweather.json
python .\scripts\run_multisplit_unseen_building.py --base-config .\configs\building_aware_unseen_itransformer_bdg1_250_noweather.json
python .\scripts\summarize_multisplit_unseen_building.py --base-run-name building_aware_unseen_itransformer_bdg1_250_noweather
```

### 5.5 输出文件位置

运行结果会写入：

- `outputs/checkpoints/`
- `outputs/results/`
- `outputs/predictions/`
- `outputs/logs/`
- `outputs/figures/`

### English

First enter the project directory:

```powershell
cd "D:\DL\BEF\Building-aware Energy Forecasting"(Please replace this with your own path)
```

### 5.1 BDG2: 250-building seen-building benchmark

Run the five training scripts and then summarize the benchmark table:

```powershell
python .\scripts\run_global_lightgbm.py
python .\scripts\run_global_lstm.py
python .\scripts\run_global_patchtst.py
python .\scripts\run_global_itransformer.py
python .\scripts\run_building_aware_itransformer.py
python .\scripts\summarize_paper_benchmark.py
```

### 5.2 BDG1: 250-building cross-dataset validation

If BDG1 has not been converted to the canonical 2015 format, run:

```powershell
python .\scripts\prepare_bdg1_2015_canonical.py --data-dir ".\Building Data Genome Project 1"
```

Then execute:

```powershell
python .\scripts\run_global_lightgbm.py --config .\configs\global_lightgbm_bdg1_250.json
python .\scripts\run_global_lstm.py --config .\configs\global_lstm_bdg1_250.json
python .\scripts\run_global_patchtst.py --config .\configs\global_patchtst_bdg1_250.json
python .\scripts\run_global_itransformer.py --config .\configs\global_itransformer_bdg1_250.json
python .\scripts\run_building_aware_itransformer.py --config .\configs\building_aware_itransformer_bdg1_250_noweather.json
python .\scripts\summarize_paper_benchmark.py --manifest .\configs\paper_benchmark_bdg1_250_manifest.json
```

### 5.3 Multi-seed stability tests

BDG2:

```powershell
python .\scripts\run_multiseed_building_aware.py
python .\scripts\summarize_multiseed_building_aware.py
```

BDG1:

```powershell
python .\scripts\run_multiseed_building_aware.py --base-config .\configs\building_aware_itransformer_bdg1_250_noweather.json
python .\scripts\summarize_multiseed_building_aware.py --base-run-name building_aware_itransformer_bdg1_250_noweather
```

### 5.4 Unseen-building boundary evaluation

BDG2:

```powershell
python .\scripts\run_unseen_building_itransformer.py
python .\scripts\run_multisplit_unseen_building.py
python .\scripts\summarize_multisplit_unseen_building.py
```

BDG1:

```powershell
python .\scripts\run_unseen_building_itransformer.py --config .\configs\building_aware_unseen_itransformer_bdg1_250_noweather.json
python .\scripts\run_multisplit_unseen_building.py --base-config .\configs\building_aware_unseen_itransformer_bdg1_250_noweather.json
python .\scripts\summarize_multisplit_unseen_building.py --base-run-name building_aware_unseen_itransformer_bdg1_250_noweather
```

### 5.5 Output locations

Generated files will be written to:

- `outputs/checkpoints/`
- `outputs/results/`
- `outputs/predictions/`
- `outputs/logs/`
- `outputs/figures/`

## 6. 当前论文主线 / Current Paper Scope

### 中文

<img width="762" height="331" alt="image" src="https://github.com/user-attachments/assets/d1f1dfc7-a861-4b83-b216-ab01253bf1ae" />

<img width="926" height="230" alt="image" src="https://github.com/user-attachments/assets/6ada7b9d-a58c-43f0-8b13-c059df2227c6" />


当前仓库围绕如下六种方法组织：

1. `Historical Average`
2. `GlobalLightGBM-building-aware`
3. `GlobalLSTM-load-only`
4. `GlobalPatchTST-load-only`
5. `GlobalITransformer-load-only`
6. `GlobalITransformer-building-aware`

其中，`GlobalITransformer-building-aware` 当前采用 **load + architecture metadata** 的设定，不启用 weather 分支，用于更清楚地研究建筑元数据本身的价值边界。

### English

The current repository is organized around the following six methods:

1. `Historical Average`
2. `GlobalLightGBM-building-aware`
3. `GlobalLSTM-load-only`
4. `GlobalPatchTST-load-only`
5. `GlobalITransformer-load-only`
6. `GlobalITransformer-building-aware`

The current `GlobalITransformer-building-aware` setup uses **load + architectural metadata** without the weather branch, so that the role of building metadata can be studied more explicitly.

## 7. GitHub Release Notes / GitHub 发布说明

### 中文

- 原始数据集默认不纳入版本控制。
- 训练生成的检查点、日志、预测结果、结果表默认不纳入版本控制。
- 本仓库已经删除历史探索代码，仅保留论文主线最小复现路径。

### English

- Raw datasets are excluded from version control by default.
- Generated checkpoints, logs, predictions, and result tables are excluded from version control.
- Historical exploratory branches have been removed so that the repository matches the current paper workflow.
