"""
Train the PET synthesis module from MRI using a transformer-based encoder-decoder.
"""

import torch
from model import SynthesisGenerator, SynthesisDiscriminator
import torch.nn as nn
import torch.optim as optim

def train_synthesis(train_loader, num_epochs=50, device="cuda"):
    generator = SynthesisGenerator().to(device)
    discriminator = SynthesisDiscriminator().to(device)
    criterion = nn.L1Loss()
    adv_loss = nn.BCELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(num_epochs):
        for batch in train_loader:
            mri, pet = batch["mri"].to(device), batch["pet"].to(device)

            synthetic_pet = generator(mri)

            optimizer_D.zero_grad()
            real_loss = adv_loss(discriminator(pet), torch.ones_like(pet))
            fake_loss = adv_loss(discriminator(synthetic_pet.detach()), torch.zeros_like(pet))
            loss_D = (real_loss + fake_loss) / 2
            loss_D.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            g_adv = adv_loss(discriminator(synthetic_pet), torch.ones_like(pet))
            g_recon = criterion(synthetic_pet, pet)
            loss_G = g_adv + g_recon
            loss_G.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss G: {loss_G.item():.4f}, Loss D: {loss_D.item():.4f}")
