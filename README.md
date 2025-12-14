# CLMedical

**Continual Learning for Histopathology Image Classification in Class-Incremental Learning**

This repository implements class-incremental continual learning methods for histopathology image classification based on the **NCT-CRC-HE-100K** dataset.  
The implementation is built on top of the **[Mammoth Continual Learning Framework](https://github.com/aimagelab/mammoth)**.

All contents of this repository are released under the **MIT License**.

---

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Project Integration](#project-integration)
- [Dataset Preparation](#dataset-preparation)
- [Preprocessing Utilities](#preprocessing-utilities)
- [Dataset Variants and Normalization Methods](#dataset-variants-and-normalization-methods)
- [Running Experiments](#running-experiments)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

- **Task**: Class-Incremental Learning (CIL)
- **Domain**: Histopathology Image Classification
- **Dataset**: NCT-CRC-HE-100K
- **Framework**: Mammoth
- **Normalization Strategies**:
  - ImageNet normalization
  - Dataset-level normalization
  - Macenko stain normalization
  - Per-image normalization

---

## Environment Setup

- **Python Version**: `Python 3.9.25`

### Install Mammoth

```bash
git clone https://github.com/aimagelab/mammoth.git
cd mammoth
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

---

## Project Integration

After downloading this repository, integrate the provided source files into the Mammoth framework.

### File Placement

| File / Folder | Destination |
|--------------|-------------|
| `compute_nct_mean_std.py` | `mammoth/data/` |
| `splitNCT/` | `mammoth/data/` |
| `preprocess_nct_macenko.py` | `mammoth/` |
| `seq_nct-224.py` | `mammoth/datasets/` |
| `seq_nct-224-allimage.py` | `mammoth/datasets/` |
| `seq_nct-224-macen.py` | `mammoth/datasets/` |
| `seq_nct-224-perimage.py` | `mammoth/datasets/` |

---

## Dataset Preparation

Download the following datasets from Google Drive:

- **NCT-CRC-HE-100K** (original dataset)
- **NCT-CRC-HE-100K-split**  
  - Train / Validation / Test split ratio: **8 : 1 : 1**
- **NCT-CRC-HE-100K-macenko-split**  
  - Macenko-normalized version of `NCT-CRC-HE-100K-split`

Place all datasets into:

```text
mammoth/data/
```

---

## Preprocessing Utilities

### Compute Dataset Mean and Standard Deviation

Compute the mean and standard deviation of the entire training set:

```bash
python compute_nct_mean_std.py
```

### Macenko Stain Normalization

Generate the Macenko-normalized dataset:

```bash
python preprocess_nct_macenko.py
```

The script will generate:

```text
NCT-CRC-HE-100K-macenko-split
```

---

## Dataset Variants and Normalization Methods

Each dataset definition corresponds to a specific normalization strategy:

| Dataset Script | Normalization Method |
|---------------|----------------------|
| `seq_nct-224.py` | ImageNet normalization |
| `seq_nct-224-allimage.py` | Dataset-level normalization |
| `seq_nct-224-macen.py` | Macenko normalization |
| `seq_nct-224-perimage.py` | Per-image normalization |

These dataset scripts also include the required adaptations for integrating the NCT dataset into the Mammoth framework.

---

## Running Experiments

After completing the configuration, experiments can be executed using the commands provided in:

```text
Experiment_Commands.docx
```

### Example Command

```bash
python main.py \
  --model derpp \
  --dataset seq-nct-224-allimage \
  --backbone resnet18 \
  --device 0 \
  --batch_size 64 \
  --n_epochs 50 \
  --alpha 0.5 \
  --beta 0.5 \
  --lr 0.03 \
  --buffer_size 500 \
  --enable_other_metrics 1
```

---

## Project Structure

```text
CLMedical/
├── compute_nct_mean_std.py
├── preprocess_nct_macenko.py
├── splitNCT/
├── datasets/
│   ├── seq_nct-224.py
│   ├── seq_nct-224-allimage.py
│   ├── seq_nct-224-macen.py
│   └── seq_nct-224-perimage.py
└── Experiment_Commands.docx
```

---

## License

This project is licensed under the **MIT License**.

