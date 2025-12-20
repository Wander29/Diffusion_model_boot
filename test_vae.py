# This project requires Python 3.10 or above:
import sys
assert sys.version_info >= (3, 10)

# We also need PyTorch ≥ 2.6.0:
from packaging.version import Version
import torch
assert Version(torch.__version__) >= Version("2.6.0")

import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torchmetrics
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, transforms

from vae_ludo.impl_vae import ConvVAE
from vae_ludo.losses import VAELoss
from ldm_ludo import training
from common import data

dataset_dir = "/home/ludo/Dev/personale/datasets/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 10
batch_size = 64
epochs = 1
lr = 1e-3

# Load dataset, split it, and load it into DataLoaders
train_data, valid_data, test_data = data.loadDataset("mnist", dataset_dir)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)

# Create the VAE model
model_vae = ConvVAE(28*28, 3, 0.3)

# let's train it
torch.manual_seed(42)
optimizer = torch.optim.NAdam(model_vae.parameters(), lr=3e-3)
# TODO metrics for what? 
loss_fn = VAELoss(0.001)
rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
history = training.train(model_vae, optimizer, loss_fn, rmse, train_loader,
                valid_loader, device=device, n_epochs=1, reconstruct=True)



