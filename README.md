# Lexeme Inflection Probing

This repository implements an end-to-end pipeline to probe transformer language models for
disentangled encoding of lexical and inflectional information, with control tasks to validate findings.

## Installation

1. Clone the repository
2. Create a virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Dataset Preparation

Dataset notebooks in `notebooks/` can be used to:

- Process UD GUM English dataset
- Generate control tasks
- Analyze data distributions

### Activation Extraction

Extract model activations for analysis using:

```bash
python -m src.activation_extraction \
    --data data/dataset.csv \
    --output output/model_dataset_reps.npz \
    --model model_name
```

### Probe Training

Two types of probing tasks are supported:

1. Multiclass inflection probing
2. Lexeme classification probing

Each comes with corresponding control tasks to validate findings.

### Running Experiments

Run complete experimental pipelines with:

```bash
python -m src.experiment \
    --model gpt2 \
    --dataset ud_gum_dataset \
    [--experiment {multiclass_inflection_dense,lexeme_dense}]
```

This will:

1. Extract activations if not already present
2. Run probing experiments with both main and control tasks
3. Perform analysis and generate visualizations

Available experiments:

- `multiclass_inflection_dense`: Dense probe for inflection classification with control task
- `lexeme_dense`: Dense probe for lexeme classification with control task

### Analysis & Visualization

The pipeline automatically generates:

- Probing accuracy plots for each layer
- Side-by-side comparisons of linguistic vs control task performance
- Layer-wise analysis including cosine similarities

Results are saved in:

```
output/probes/{dataset}_{model}_{experiment_name}/
output/{model}_{dataset}_analysis/
```

## Project Structure

```
src/
  ├── activation_extraction.py  # Extract model hidden states
  ├── train.py                 # Train probing classifiers
  ├── analysis.py             # Analysis and visualization
  ├── experiment.py           # End-to-end experimental pipeline
  ├── config.py              # Configuration settings
  └── utils.py                # Helper functions
notebooks/
  ├── dataset.ipynb          # Dataset creation and analysis
  └── analysis.ipynb         # Results analysis and plotting
```
