"""
Joint training of IMAGINE-Net: PET synthesis and classification using MRI input.
"""

import torch
from model import IMAGINENet
import torch.nn as nn
import torch.optim as optim

def train_joint(train_loader, num_epochs=30, device="cuda"):
    model = IMAGINENet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        for batch in train_loader:
            mri, label = batch["mri"].to(device), batch["label"].float().to(device)

            output = model(mri)
            loss = criterion(output.squeeze(), label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
