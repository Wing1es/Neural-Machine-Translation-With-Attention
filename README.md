# Neural Machine Translation (NMT) with Attention QKV

This repository contains a Jupyter notebook implementing a Neural Machine Translation (NMT) model using the Query-Key-Value (QKV) attention mechanism.

---

## Overview

- Implements a transformer-based NMT model with attention.
- Demonstrates training and inference within a single notebook.
- Uses temperature-based sampling for token generation.
- Built with [Trax](https://github.com/google/trax) and JAX for efficient model building and training.

---

## Getting Started

### Requirements

- Python 3.8+
- Jupyter Notebook
- Trax
- JAX
- NumPy

Install dependencies:

bash
pip install trax jax numpy jupyter

How to Run
Clone this repository.

Launch Jupyter Notebook:

bash
Copy
Edit
jupyter notebook
Open the notebook file NMT_Attention_QKV.ipynb.

Run the cells sequentially to train the model and generate translations.

Notebook Contents
Data loading and preprocessing.

Model definition with Attention QKV.

Training loop setup.

Token-by-token inference with temperature sampling.

Evaluation snippets.

Notes
This notebook is self-contained; all code for model training and inference is inside.

For larger datasets or more complex workflows, consider separating code into scripts.

License
This project is licensed under the MIT License.
