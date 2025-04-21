# Probing Lexemes and Inflections

Minimal steps to extract activations, train probes, and analyze.

## 1. Install

1. Clone the repo
2. Create a virtual environment and install the dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Extract activations

```bash
python -m src.activation_extraction \
  --data data/your_dataset.csv \
  --output-dir output/<model>_reps \
  --model <model_key>
```

## 3. Train probes

```bash
python -m src.train \
  --activations output/<model>_reps \
  --labels data/your_dataset.csv \
  --task [lexeme|multiclass_inflection|binary_inflection] \
  --dataset your_dataset \
  --exp_label run1 \
  --layers 0,1,2       # optional
  --probe_type [linear|nn]
```

Results + plots → `output/probes/`

## 4. Unsupervised analysis

```bash
python -m src.analysis \
  --activations-dir output/<model>_reps \
  --labels data/your_dataset.csv \
  --model <model_key> \
  --dataset your_dataset
```

Outputs → `output/<model>_analysis/`

---

Adjust `src/config.py` for model settings and hyperparameters.

```

```
