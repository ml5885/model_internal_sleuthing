# Lexeme Inflection Probing

This repository implements an end-to-end pipeline to probe transformer language models for
disentangled encoding of lexical (verb identity) and inflectional (tense) information.

## Installation

1. Clone the repository.
2. Create a virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Activation Extraction

Run the extraction module on a subset or the full dataset:

```bash
python -m src.activation_extraction --data data/controlled_sentences.csv --output output/gpt2_reps.npz --model gpt2
```

### Probe Training

Train linear probes on the saved activations:

```bash
python -m src.train --activations output/gpt2_reps.npz --labels data/controlled_sentences.csv --task both
```

The `--task` flag can be set to `tense`, `lexeme`, or `both`. The script uses PyTorch and runs training over each layer separately, saving performance metrics and confusion matrices.

### Unsupervised Analysis

Perform KMeans clustering and cosine similarity analysis across layers:

```bash
python -m src.analysis --activations output/gpt2_reps.npz --labels data/controlled_sentences.csv
```

## HPC / SLURM

Use the provided shell scripts in the `scripts/` folder to run jobs on your cluster. For example:

```bash
bash scripts/slurm_submission_example.sh
```

## Testing

Run all tests via pytest:

```bash
pytest tests
```

## Extensibility

The project is designed to be model-agnostic. To add a new model, update `src/config.py` and extend or subclass the interface in `src/model_wrapper.py`.

Happy probing!
