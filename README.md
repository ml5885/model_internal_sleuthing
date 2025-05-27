## Overview

This codebase implements a pipeline for extracting hidden activations from language models and training classifiers on those activations to predict lexeme and inflectional features. It also provides scripts for unsupervised analysis, intrinsic dimensionality (PCA), and analogy completion experiments.

---

## Structure

- **src/**: Main pipeline and utilities

  - `activation_extraction.py`: Extracts hidden states from models for each token in the dataset.
  - `train.py`: Trains classifiers (linear, MLP and random forest) to predict lexeme or inflection labels from activations.
  - `probe.py`: Defines the classifiers (linear, MLP and random forest)
  - `analysis.py`: Runs unsupervised analyses (t-SNE, clustering, cosine similarity) on activations.
  - `experiment.py`: Runs the full pipeline (extraction, probing, analysis).
  - `pca_experiment.py`: Computes PCA/intrinsic dimensionality of activations.
  - `analogy_completion.py`: Runs word analogy completion tasks using input embeddings.
  - `config.py`: Model settings and hyperparameters.
  - 
- **dataset/**: Utilities and notebooks for dataset construction and statistics.

  - `dataset.ipynb`: Builds the probing dataset from UD English-GUM.
  - `dataset_statistics.py`: Computes and plots dataset statistics.

- **plots/**: Scripts and figures for plotting and analysis results.

  - `plot_classifier_results.py`, `plot_additional_results.py`, `plot_pca_results.py`, `small_results_plot.py`: Plotting and analysis scripts.
  - `scratch2.py`, `scratch3.ipynb`: Miscellaneous scripts and exploratory notebooks.
  - `delta_rank_bar.png`, `tokenize_vs_sum_scatter_old.png`, and folders like `figs/`, `figures/`, etc.: Generated figures and plots.

---

## Running various experiments

**Extract activations**

   ```bash
   python -m src.activation_extraction \
     --data data/your_dataset.csv \
     --output-dir output/<model>_reps \
     --model <model_key>
   ```

   - `--data`: CSV with 'Sentence' and 'Target Index'
   - `--output-dir`: Directory for activation shards
   - `--model`: Model key (see `src/config.py`)

**Train classifiers**

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

**Unsupervised analysis**

   ```bash
   python -m src.analysis \
     --activations-dir output/<model>_reps \
     --labels data/your_dataset.csv \
     --model <model_key> \
     --dataset your_dataset
   ```

   - Outputs are in `output/<model>_analysis/`

**Additional analyses**

   - **Intrinsic dimensionality (PCA):**  
     Run `src/pca_experiment.py` or use `scripts/run_pca_experiments.sh` to analyze the dimensionality of representations.
   - **Analogy completion:**  
     Use `src/analogy_completion.py` or the provided scripts to evaluate analogy-solving ability of input embeddings.
   - **Plotting and statistics:**  
     Use scripts in `plots/` for probe result plots, PCA/analogy plots, and selectivity/advantage analyses. Dataset statistics are in `dataset/`.

Edit `src/config.py` to add or modify models, hyperparameters, or checkpoint lists.
