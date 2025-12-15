import matplotlib.pyplot as plt

import torch
from datetime import datetime
import glob
import os

def setup_plot_font_sizes():
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

def plot_multiple_images(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plot_image(image)

def plot_image(image):
    plt.imshow(image.permute(1, 2, 0).cpu(), cmap="binary")
    plt.axis("off")


def save_model(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20251215_142530
    filename = f"weights/diff_model_{timestamp}.pt"
    torch.save(diffusion_model.state_dict(), filename)

def load_last_model(model):
    files = glob.glob("weights/diff_model_*.pt")
    latest = max(files, key=os.path.getctime)
    state = torch.load(latest, map_location="cpu")
    model.load_state_dict(state)