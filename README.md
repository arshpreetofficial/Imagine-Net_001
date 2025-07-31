# IMAGINE-Net_001

This repository contains the implementation for the paper:
**"IMAGINE-Net: A Unified Deep Learning Framework for Predicting MCI to AD Conversion Using Incomplete Neuroimaging Data"**

## Getting Started

### Requirements

- Python >= 3.8  
- PyTorch >= 1.10  
- NumPy, SciPy, scikit-learn  
- Matplotlib, Seaborn  
- nibabel (for MRI data)  
- tqdm, pandas
## Directory Structure
IMAGINE-Net_001/
├── config.yaml
├── README.md
├── requirements.txt
├── code/
│   ├── evaluate.py
│   ├── model.py
│   ├── train_joint.py
│   └── train_synthesis.py
├── data/
│   └── README.md
├── figures/
│   ├── bar_chart.png
│   ├── epoc.png
│   ├── Proposed.png
│   ├── roc_comparision.png
│   └── workflow.png
├── notebooks/
│   ├── Ablation_Study.ipynb
│   ├── IMAGINENet_Demo.ipynb
│   ├── Load_Pretrained.ipynb
│   ├── Train_Evaluate.ipynb
│   └── Visualize_Results.ipynb
├── results/
│   ├── curves/
│   │   ├── pr_curve_clas.png
│   │   ├── roc_curve_adni.png
│   │   └── training_loss_plot.png
│   └── inference_output/
│       └── predictions.csv

## Instructions

1. Download the required datasets (ADNI, OASIS, AIBL) and place symbolic links or downloaded files in `data/`.
2. Modify `config.yaml` as needed.
3. Run training: `python code/train.py`
4. Run evaluation: `python code/evaluate.py`

## Datasets

Please request or download the following datasets:

- [ADNI](https://adni.loni.usc.edu/)
- [OASIS](https://sites.wustl.edu/oasisbrains/)
- [AIBL](https://aibl.csiro.au/)


