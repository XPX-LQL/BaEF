# BaEF: Building-aware Global ITransformer Framework for Multi-Building Energy Forecasting

This repository contains the cleaned research codebase for our BaEF paper line:

**From traditional Building Energy Forecasting (BEF) to Building-aware Energy Forecasting (BaEF), with a 250-building benchmark on BDG2 and BDG1, and a GlobalITransformer-building-aware framework for studying the boundary of building metadata under seen-building and unseen-building settings.**

## 1.  Hardware and Software

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

## 2.  Code Structure

```text
Building-aware Energy Forecasting/

├── configs/                                
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
├── scripts/                                
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
├── src/building_aware/                     
│   ├── baselines.py
│   ├── data_utils.py
│   ├── features.py
│   ├── global_lightgbm.py
│   ├── itransformer.py
│   ├── lstm.py
│   ├── metrics.py
│   ├── patchtst.py
│   └── time_splits.py
├── outputs/                                
├── .gitignore
├── README.md
└── requirements.txt
```

The current GitHub version keeps only the paper-critical and minimally reproducible workflow. Historical exploratory branches, cache folders, and generated training artifacts have been removed.

Core directories:

- `configs/`: experiment configurations used in the paper workflow.
- `scripts/`: training, evaluation, and result summarization entry points.
- `src/building_aware/`: core implementations for data loading, feature engineering, baselines, iTransformer, PatchTST, metrics, and time splitting.
- `outputs/`: runtime outputs such as checkpoints, predictions, logs, and result tables.

## 3.  Dataset Download

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

## 4.  Environment Setup

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

## 5.  Training and Evaluation

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

## 6.  Current Paper Scope


<img width="762" height="331" alt="image" src="https://github.com/user-attachments/assets/d1f1dfc7-a861-4b83-b216-ab01253bf1ae" />

<img width="926" height="230" alt="image" src="https://github.com/user-attachments/assets/6ada7b9d-a58c-43f0-8b13-c059df2227c6" />

The current repository is organized around the following six methods:

1. `Historical Average`
2. `GlobalLightGBM-building-aware`
3. `GlobalLSTM-load-only`
4. `GlobalPatchTST-load-only`
5. `GlobalITransformer-load-only`
6. `GlobalITransformer-building-aware`

The current `GlobalITransformer-building-aware` setup uses **load + architectural metadata** without the weather branch, so that the role of building metadata can be studied more explicitly.

## 7.  GitHub Release Notes

### English

- Raw datasets are excluded from version control by default.
- Generated checkpoints, logs, predictions, and result tables are excluded from version control.
- Historical exploratory branches have been removed so that the repository matches the current paper workflow.
