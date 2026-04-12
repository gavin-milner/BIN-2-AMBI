# 🎧 Binaural to Ambisonics Conversion – Training Pipeline

This repository contains the **training component** of a binaural to ambisonics conversion pipeline developed as part of a final year MAI thesis in Electronic and Computer Engineering at Trinity College Dublin.

The focus of this repository is on the **model training and experimental pipeline** used in the project.

---

## 📌 Overview

This project explores the conversion of **binaural audio signals into ambisonic representations** using machine learning techniques.

The repository includes:
- Training scripts and pipelines  
- Model architectures  
- Configuration files  
- Binaural Analysis preprocessing utilities  

---

## 📝 Repository Scope

This is a **public release** of the original project.

The following have been removed:
- Sensitive or proprietary code 
- Private configurations and internal tooling  
- Any non-public data  

The repository preserves the **core training workflow and evaluation methodology** used in the thesis.

---

## 📂 Project Structure

```
.
├── Architectures/          # Model architecture definitions (C-RNN variants, CNNs, ablations)
├── Configs/                # YAML configuration files for analysis
├── Datasets/
│   ├── listening_test/     # Listening test scenes with features, FFT, and ground truth
│   └── ...                 # Additional datasets (not included in public release)
├── Experiments/
│   ├── 6000scenes_no_bg/   # Main experiment results, plots, and significance tests
│   ├── CRNN_V15_*/         # Model training runs with checkpoints and predictions
│   ├── Perceptual_*/       # Perceptual loss experiments (MAE/MSE, with/without BG)
│   └── ...                 # Additional experiments (not included in public release)
├── Modules/
│   ├── AMBIQUAL/           # (external) Ambisonics quality metric
│   ├── AmbiScaper/         # (external) Ambisonics scene generation
│   ├── BINASPECT/          # (external) Binaural audio analysis
│   └── DirAC_Synthesis/    # Parametric spatial audio synthesis
|── resources/
│   ├── HRTF/               # Head related transfer functions
│   ├── ht_captures/        # Head rotation signals
│   ├── ht_synthetic/       # Synthetically generated head rotation signals
├── src/
│   ├── Analysis/           # Feature extraction and analysis pipeline
│   ├── Evaluation/         # AMBIQUAL evaluation scripts
│   ├── Training/           # Training notebook (BIN2AMBI)
│   └── utils/              # Plotting, significance testing, and helper utilities
├── requirements.txt
└── README.md
```

> **Note:** Modules marked as *(external)* are third-party repositories not included in this release.

| Module | Repository |
|--------|-----------|
| AMBIQUAL | https://github.com/QxLabIreland/Ambiqual |
| AmbiScaper | https://github.com/Nilson/ambiscaper |
| BINASPECT | https://github.com/QxLabIreland/Binaspect |

