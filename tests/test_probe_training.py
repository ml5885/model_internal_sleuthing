import numpy as np
from src import train

def test_split_data():
    print("Starting test: test_split_data")
    # Generate random dataset
    X = np.random.randn(100, 768)
    y = np.random.randint(0, 2, size=(100,))
    print("Generated dummy data for X and y.")
    
    # Split the data using the provided function
    X_train, y_train, X_val, y_val, X_test, y_test = train.split_data(X, y)
    print("Data split into training, validation, and testing sets.")
    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    
    print(f"Total samples after split: {total} (Expected: 100)")
    assert total == 100, f"Expected total of 100 samples, got {total}"
    print("test_split_data passed.\n")

def test_train_runs():
    print("Starting test: test_train_runs")
    n_samples = 50
    # Generate random data
    X = np.random.randn(n_samples, 768)
    y = np.random.randint(0, 2, size=(n_samples,))
    print("Generated dummy data for training.")

    # Split the data
    X_train, y_train, X_val, y_val, X_test, y_test = train.split_data(X, y)
    print("Data split completed for train, validation, and test sets.")

    # Train the probe model
    print("Training the probe model...")
    model = train.train_probe(X_train, y_train, X_val, y_val, 768, 2)
    print("Training completed. Evaluating the model...")

    # Evaluate the model
    acc, cm = train.evaluate_probe(model, X_test, y_test)
    print(f"Evaluation results: Accuracy = {acc}, Confusion Matrix =\n{cm}")

    assert 0.0 <= acc <= 1.0, f"Accuracy out of bounds: {acc}"
    print("test_train_runs passed.\n")
