import matplotlib.pyplot as plt

def plot_2d_latent_space(latents_2d, labels):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                        c=labels, cmap='coolwarm', s=10, alpha=0.7)

    handles, _ = scatter.legend_elements()
    custom_labels = ['Not Fraud', 'Fraud']
    legend1 = plt.legend(handles, custom_labels, title="Classes")
    plt.gca().add_artist(legend1)

    plt.title("Latent Space")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.show()

def plot_3d_latent_space(latents_3d, labels):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    scatter = ax.scatter(latents_3d[:, 0], latents_3d[:, 1], latents_3d[:, 2],
                        c=labels, cmap='coolwarm', s=15, alpha=0.7)

    # Optional: Add custom legend
    handles, _ = scatter.legend_elements()
    custom_labels = ['Not Fraud', 'Fraud']
    legend1 = ax.legend(handles, custom_labels, title="Classes")
    ax.add_artist(legend1)

    # Label axes
    ax.set_title("3D Latent Space")
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    ax.set_zlabel("Latent Dim 3")

    plt.show()
