"""
Defines the IMAGINE-Net architecture for image synthesis and classification.
"""

import torch.nn as nn

class SynthesisGenerator(nn.Module):
    def __init__(self):
        super(SynthesisGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class SynthesisDiscriminator(nn.Module):
    def __init__(self):
        super(SynthesisDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(1, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class IMAGINENet(nn.Module):
    def __init__(self):
        super(IMAGINENet, self).__init__()
        self.generator = SynthesisGenerator()
        self.classifier = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        synthetic_pet = self.generator(x)
        return self.classifier(synthetic_pet)
