import torch
import numpy as np

from datetime import datetime
import glob
import os
import random

# - Models
def save_model(model, prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20251215_142530
    filename = f"weights/{prefix}_{timestamp}.pt"
    torch.save(model.state_dict(), filename)

def load_last_model(model):
    files = glob.glob("weights/{prefix}_*.pt")
    latest = max(files, key=os.path.getctime)
    state = torch.load(latest, map_location="cpu")
    model.load_state_dict(state)

# - General
def set_seed(seed: int = 111):
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False