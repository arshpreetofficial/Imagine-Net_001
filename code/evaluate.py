"""
Evaluation script for IMAGINE-Net model.
"""

import torch
from sklearn.metrics import roc_auc_score
from model import IMAGINENet

def evaluate(model, test_loader, device="cuda"):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            mri, label = batch["mri"].to(device), batch["label"].float().to(device)
            output = model(mri)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(output.squeeze().cpu().numpy())

    auc = roc_auc_score(y_true, y_pred)
    print(f"AUC: {auc:.4f}")
