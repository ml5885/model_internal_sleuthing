import numpy as np
import torch
import pytest

from src.probe import closed_form_ridge_binary_predict, process_inflection_layer

def test_closed_form_ridge_binary_predict_line():
    # y = 2*x  perfectly linear
    X = torch.tensor([[1.],[2.],[3.]], dtype=torch.float32)
    y = torch.tensor([2.,4.,6.], dtype=torch.float32)
    pred = closed_form_ridge_binary_predict(X, y, X, lambda_reg=0.0)
    # should recover exactly
    assert torch.allclose(pred.squeeze(), y, atol=1e-4)

def test_process_inflection_layer_perfectly_separable():
    # build a small toy layer: 4 points in 2D, two classes
    # class 0 → cluster at (1,1), class1 → cluster at (-1,-1)
    X = np.array([[1,1],[1,1],[-1,-1],[-1,-1]], dtype=float)
    labels = np.array([0,0,1,1])
    # target_indices is ignored for 2D input so just zeros
    tgt = np.zeros(4, dtype=int)
    layer_id, res = process_inflection_layer(
        layer=5,
        X_layer=X,
        inflection_labels=labels,
        lambda_reg=1e-3,
        target_indices=tgt
    )
    assert layer_id == 5
    # perfect separation ⇒ acc ≈ 1.0
    assert pytest.approx(res["inflection_acc"], rel=1e-3) == 1.0
    # control should be between 0 and 1
    assert 0.0 <= res["inflection_control_acc"] <= 1.0
