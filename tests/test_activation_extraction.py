import os
import numpy as np
import pandas as pd
import tempfile
from src import activation_extraction, config

def test_extraction_output_shape():
    # Create a temporary CSV with sample data
    df = pd.DataFrame({
        "Sentence": ["I ate in the past.", "I eat every day."],
        "Target Index": [2, 2],
        "Lemma": ["eat", "eat"],
        "Inflection Label": ["past", "present"],
        "Word Form": ["ate", "eat"],
        "Category": ["Verb", "Verb"],
        "Dimension": ["Tense/Aspect", "Tense/Aspect"],
        "Source Type": ["Template", "Template"]
    })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        data_file = tmp.name

    output_file = tempfile.NamedTemporaryFile(suffix=".npz", delete=False).name

    activation_extraction.extract_and_save(data_file, output_file, "gpt2")
    out = np.load(output_file)
    activations = out["activations"]
    # For two samples, expect shape: (2, n_layers, d_model)
    assert activations.shape[0] == 2
    assert len(activations.shape) == 3

    os.remove(data_file)
    os.remove(output_file)
