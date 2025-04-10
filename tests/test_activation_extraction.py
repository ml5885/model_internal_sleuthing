import os
import numpy as np
import pandas as pd
import tempfile
from src import activation_extraction, config

def test_extraction_output_shape():
    print("Starting test: test_extraction_output_shape")
    # Create a temporary CSV with sample data
    print("Creating sample DataFrame with known values...")
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

    # Saving CSV to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        data_file = tmp.name
    print(f"Temporary CSV created: {data_file}")

    # Define a temporary output file for the npz activations
    output_file = tempfile.NamedTemporaryFile(suffix=".npz", delete=False).name
    print(f"Temporary output file will be written: {output_file}")

    # Run the activation extraction
    print("Calling activation_extraction.extract_and_save with model 'gpt2' ...")
    activation_extraction.extract_and_save(data_file, output_file, "gpt2")

    # Load and validate the output
    print("Loading output npz file...")
    out = np.load(output_file)
    activations = out["activations"]
    print(f"Activations loaded. Shape received: {activations.shape}")

    # For two samples, expect shape: (2, n_layers, d_model)
    assert activations.shape[0] == 2, f"Expected 2 samples, got {activations.shape[0]}"
    assert len(activations.shape) == 3, f"Expected 3 dimensions (samples, layers, d_model), got {len(activations.shape)}"

    # Clean up temporary files
    os.remove(data_file)
    os.remove(output_file)
    print("Temporary files removed and test_extraction_output_shape passed.\n")
