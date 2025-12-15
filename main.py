# This project requires Python 3.10 or above:
import sys
assert sys.version_info >= (3, 10)

# We also need PyTorch ≥ 2.6.0:
from packaging.version import Version
import torch
assert Version(torch.__version__) >= Version("2.6.0")

from torch.utils.data import DataLoader
import torch.nn as nn
import torchmetrics
# from collections import namedtuple

from ldm_ludo import diff_model as dm
from ldm_ludo import data
from ldm_ludo import plots
from ldm_ludo import utils
from ldm_ludo import training

# Prefer and hw accelerator
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Set seed for reproducibility
torch.manual_seed(42)

# Variance schedule: get alphas and betas
T = int(10)
embed_dim = 64 # TODO change this for time embedding
alphas, betas, alpha_bars = dm.variance_schedule(T)

"""In the DDPM paper, the authors used $T = 1,000$, while in the Improved DDPM, they bumped this up to $T = 4,000$, so we use this value. 
The variables `alphas`, `betas`, and `alpha_bars` contain $\alpha_t$, $\beta_t$, $\bar{\alpha}_t$ respectively, starting from _t_ = 0.
"""

# Load dataset, split it, and load it into DataLoaders
train_data, valid_data, test_data = data.loadDataset("mnist")
train_set = data.DiffusionDataset(train_data, T, alpha_bars)  # wrap dataset
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_set = data.DiffusionDataset(valid_data, T, alpha_bars)
valid_loader = DataLoader(valid_set, batch_size=32)

# let's train the model
torch.manual_seed(42)
diffusion_model = dm.DiffusionModel(T).to(device)
huber = nn.HuberLoss()
optimizer = torch.optim.NAdam(diffusion_model.parameters(), lr=3e-3)
rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
history = training.train(diffusion_model, optimizer, huber, rmse, train_loader,
                valid_loader, device=device, n_epochs=1)

# save model's trained weights
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20251215_142530
filename = f"diff_model_{timestamp}.pt"
torch.save(diffusion_model.state_dict(), filename)

import glob
import os

files = glob.glob("model_*.pt")
latest = max(files, key=os.path.getctime)  # or use os.path.getmtime
state = torch.load(latest, map_location="cpu")
model.load_state_dict(state)

# Generate images
X_gen = dm.generate_ddpm(diffusion_model)  # generated images
utils.plot_multiple_images(X_gen, 8)
plt.show()

# use DDIM sampling
X_gen_ddim = dm.generate_ddim(diffusion_model, num_steps=500)
utils.plot_multiple_images(X_gen_ddim, 8)
plt.show()