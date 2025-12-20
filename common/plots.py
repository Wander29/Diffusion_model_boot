import torch
import matplotlib.pyplot as plt

# - Plots
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

def plot_learning_curves(history):
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_images_mnist(images, labels=None):
    with torch.no_grad():
        n_cols = 8
        n_rows = (len(images) - 1) // n_cols + 1
        plt.figure(figsize=(n_cols, n_rows))
        fig, axes = plt.subplots(1, 8, figsize=(n_cols, n_rows))
        for i, ax in enumerate(axes):
            img = images[i].squeeze(0).cpu().numpy()  # (H,W)
            if (labels is not None):
                ax.set_title(str(int(labels[i])))
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')