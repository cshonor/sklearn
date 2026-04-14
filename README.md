# Hands-On Machine Learning (2nd ed.) — Notes & Code

## English

This repository contains my learning notes and code implementations for the book **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow (2nd Edition)**.

## 中文

本仓库包含《Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow（第二版）》的学习笔记与代码实现。

## What’s inside

- **01-machine-learning-basics/**: Traditional ML with Scikit-Learn (Ch1–9)
- **02-neural-networks-deep-learning/**: Deep Learning with TF/Keras (Ch10–19)
- Each chapter folder contains:
  - **notes/**: notes (can be multiple `.md`)
  - **code/**: runnable code examples
  - **images/**: chapter-specific images/plots
- **datasets/**: shared datasets (optional)
- **assets/**: shared images/plots (optional)

## Environment

- Python 3.8+
- Recommended: **Anaconda** / Miniconda

### Anaconda (recommended)

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate sklearn
```

Then run code inside this environment.

### pip

If you don't use conda:

```bash
pip install numpy scipy matplotlib scikit-learn
```

Or use `requirements.txt`:

```bash
pip install -r requirements.txt
```

## References

- scikit-learn docs: `http://scikit-learn.org/stable/index.html`

## Repository structure

```
sklearn/
├── README.md
├── environment.yml
├── requirements.txt
├── 01-machine-learning-basics/
│   ├── ch01-machine-learning-overview/
│   ├── ...
│   └── ch09-unsupervised-learning/
├── 02-neural-networks-deep-learning/
│   ├── ch10-keras-ann-intro/
│   ├── ...
│   └── ch19-large-scale-training-deployment/
├── datasets/  # optional
└── assets/    # optional
```

## Run an example

```bash
python 01-machine-learning-basics/ch06-decision-trees/code/decision_tree_animals.py
```
