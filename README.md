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

Extract activations from a model using:

```bash
python -m src.activation_extraction \
    --data data/controlled_sentences.csv \
    --output output/gpt2_reps.npz \
    --model gpt2
```

### Probe Training

Train linear probes on the saved activations:

```bash
python -m src.train \
    --activations output/gpt2_reps.npz \
    --labels data/controlled_sentences.csv \
    --task multiclass_inflection \
    [--sparse_k K] \
    [--control_inflection] \
    [--control_lexeme] \
    [--one_vs_rest]
```

The `--task` parameter accepts:

- `multiclass_inflection`: For tense classification
- `binary_inflection`: For binary tense classification
- `lexeme`: For verb identity classification

Optional arguments:

- `--sparse_k`: Enable k-sparse probing with specified k
- `--control_inflection`: Run control experiment for inflection probe
- `--control_lexeme`: Run control experiment for lexeme probe
- `--one_vs_rest`: Use one-vs-rest classification strategy

### Unsupervised Analysis

Perform layer-wise analysis including cosine similarities and clustering:

```bash
python -m src.analysis \
    --activations output/gpt2_reps.npz \
    --labels data/controlled_sentences.csv \
    --model gpt2 \
    --dataset controlled
```

### Running Full Experiments

For convenience, you can run complete experimental pipelines using:

```bash
python -m src.experiment [arguments]
```

## Testing

Run all tests via pytest:

```bash
pytest tests
```
