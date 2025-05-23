# Finding Lexical Identity and Inflectional Morphology in Modern Language Models

This repository contains the code for the paper **Model Internal Sleuthing: Finding Lexical Identity and Inflectional Morphology in Modern Language Models**.

---

## Overview

This codebase implements a pipeline for extracting hidden activations from language models and training classifiers on those activations to predict lexeme and inflectional features. It also provides scripts for unsupervised analysis, intrinsic dimensionality (PCA), and analogy completion experiments.

---

## Structure

- **src/**: Main pipeline and utilities

  - `activation_extraction.py`: Extracts hidden states from models for each token in your dataset.
  - `train.py`: Trains probes (linear, MLP, random forest, etc.) to predict lexeme or inflection labels from activations.
  - `analysis.py`: Runs unsupervised analyses (t-SNE, clustering, cosine similarity) on activations.
  - `experiment.py`: Runs the full pipeline (extraction, probing, analysis).
  - `pca_experiment.py`: Computes PCA/intrinsic dimensionality of activations.
  - `analogy_completion.py`: Runs word analogy completion tasks using input embeddings.
  - `config.py`: Model settings and hyperparameters.

- **scripts/**: Shell scripts for running experiments and analyses.

- **dataset/**: Utilities and notebooks for dataset construction, statistics, and plotting.

  - `dataset.ipynb`: Builds the probing dataset from UD English-GUM.
  - `dataset_statistics.py`: Computes and plots dataset statistics.
  - `plot_classifier_results.py`, `plot_additional_results.py`, `small_results_plot.py`: Various plotting and analysis scripts.
  - `scratch2.py`, `scratch3.ipynb`: Miscellaneous scripts and exploratory notebooks.

- **output/**: Results, probe outputs, and analysis figures are saved here.

---

## How to Run

1. **Install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Extract activations**

   ```bash
   python -m src.activation_extraction \
     --data data/your_dataset.csv \
     --output-dir output/<model>_reps \
     --model <model_key>
   ```

   - `--data`: CSV with 'Sentence' and 'Target Index'
   - `--output-dir`: Directory for activation shards
   - `--model`: Model key (see `src/config.py`)

3. **Train probes**

   ```bash
   python -m src.train \
     --activations output/<model>_reps \
     --labels data/your_dataset.csv \
     --task [lexeme|multiclass_inflection|binary_inflection] \
     --dataset your_dataset \
     --exp_label run1 \
     --layers 0,1,2       # optional
     --probe_type [reg|mlp|nn|rf]
   ```

   - Results and plots are saved in `output/probes/`

4. **Unsupervised analysis**

   ```bash
   python -m src.analysis \
     --activations-dir output/<model>_reps \
     --labels data/your_dataset.csv \
     --model <model_key> \
     --dataset your_dataset
   ```

   - Outputs are in `output/<model>_analysis/`

5. **Additional analyses**

   - **Intrinsic dimensionality (PCA):**  
     Run `src/pca_experiment.py` or use `scripts/run_pca_experiments.sh` to analyze the dimensionality of representations.
   - **Analogy completion:**  
     Use `src/analogy_completion.py` or the provided scripts to evaluate analogy-solving ability of input embeddings.
   - **Plotting and statistics:**  
     Use scripts in `dataset/` for dataset statistics, probe result plots, and selectivity/advantage analyses.

---

## Customization

- Edit `src/config.py` to add or modify models, hyperparameters, or checkpoint lists.

---

## Output

- `output/<model>_reps/`: Activation shards for each model.
- `output/probes/`: Probe results and plots.
- `output/<model>_analysis/`: Unsupervised analysis outputs.
- `notebooks/figures*/`: Figures and tables from PCA/analogy experiments.

---

## Citation

If you use this codebase, please cite the paper appropriately.
