import torch
import torch.nn as nn
import math
from src import config, utils
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

def get_device():
    return torch.device("cpu")

class LinearProbe(nn.Module):
    """A standard linear probe."""
    def __init__(self, input_dim, output_dim):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

class KSparseLinearProbe(nn.Module):
    """
    A k-sparse linear probe that retains only the top-k coefficients (by absolute value)
    in each row of the weight matrix after each optimizer step.
    
    Reference:
        Gurnee et al. (2023). Finding Neurons in a Haystack: Case Studies with Sparse Probing.
    """
    def __init__(self, input_dim: int, output_dim: int, k: int):
        super(KSparseLinearProbe, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.k = k
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)
    
    def project_weights(self):
        with torch.no_grad():
            for i in range(self.weight.size(0)):
                row = self.weight[i]
                if self.k < row.numel():
                    _, idx = torch.topk(torch.abs(row), self.k)
                    mask = torch.zeros_like(row)
                    mask[idx] = 1.0
                    self.weight[i] *= mask

def train_probe(X_train, y_train, X_val, y_val, input_dim, output_dim, lambda_reg=1e-3, quiet=True):
    torch.manual_seed(config.SEED)
    device = get_device()
    
    model = LinearProbe(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN_PARAMS["learning_rate"],
                                 weight_decay=config.TRAIN_PARAMS["weight_decay"])
    
    epochs = config.TRAIN_PARAMS["epochs"]
    best_val_acc = 0.0
    best_model_state = None
    
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            acc = accuracy_score(y_val.cpu().numpy(), preds)
            utils.log_info(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Val Acc={acc:.4f}")
            if acc > best_val_acc:
                best_val_acc = acc
                best_model_state = model.state_dict().copy()
    
    if best_model_state is None:
        best_model_state = model.state_dict().copy()
    
    model.load_state_dict(best_model_state)
    return model

def evaluate_probe(model, X_test, y_test):
    device = get_device()
    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    
    y_true = y_test.cpu().numpy()
    if set(y_true).issubset({0, 1}):
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
    else:
        cm = confusion_matrix(y_true, preds)
    
    acc = accuracy_score(y_true, preds)
    return acc, cm
