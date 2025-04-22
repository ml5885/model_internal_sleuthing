import numpy as np
import os
import tempfile
import pytest

from src.analysis import load_activations, avg_group_cosine

def make_npz(path, arr):
    np.savez_compressed(path, activations=arr)

def test_load_activations_single_file(tmp_path):
    arr = np.random.randn(5, 3, 4)
    f = tmp_path / "a.npz"
    make_npz(str(f), arr)
    out = load_activations(str(f))
    assert out.shape == arr.shape
    assert np.allclose(out, arr)

def test_load_activations_directory(tmp_path):
    arr1 = np.ones((2,3,4))
    arr2 = np.zeros((1,3,4))
    f1 = tmp_path / "activations_part0.npz"
    f2 = tmp_path / "activations_part1.npz"
    make_npz(str(f1), arr1)
    make_npz(str(f2), arr2)
    out = load_activations(str(tmp_path))
    # should stack along axis=0
    assert out.shape == (3,3,4)
    assert np.allclose(out[:2], arr1)
    assert np.allclose(out[2:], arr2)

def test_avg_group_cosine_perfect_and_orthogonal():
    # group 0: [1,0],[1,0] → cosine=1
    # group 1: [0,1],[0,1] → cosine=1
    acts = np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
    labels = np.array([0,0,1,1])
    ci = avg_group_cosine(acts, labels)
    assert pytest.approx(ci, rel=1e-6) == 1.0

def test_avg_group_cosine_mixed():
    # group 0: identical unit vectors → 1
    # group 1: opposite vectors → -1
    acts = np.array([[1.,0.],[1.,0.],[-1.,0.],[1.,0.]])
    labels = np.array([0,0,1,1])
    ci = avg_group_cosine(acts, labels)
    # group0 mean=1, group1 mean=cos(-1,1)= -1 so average=0
    assert pytest.approx(ci, rel=1e-6) == 0.0
