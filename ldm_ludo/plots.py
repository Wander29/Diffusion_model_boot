import matplotlib.pyplot as plt
import torch 

import utils

def plot_variance_schedule(alpha_bars, betas, T):
    # extra code – this cell generates Figure 18–21
    plt.figure(figsize=(6, 3))
    plt.plot(betas, "r--", label=r"$\beta_t$")
    plt.plot(alpha_bars, "b", label=r"$\bar{\alpha}_t$")
    plt.axis([0, T, 0, 1.01])
    plt.grid(True)
    plt.xlabel("$t$")
    plt.ylabel(r"Value")
    plt.legend()
    plt.show()

def original_image(alpha_bars, sample, noise):
    alpha_bars_t = torch.gather(alpha_bars, dim=0, index=sample.t.squeeze(1))
    alpha_bars_t = alpha_bars_t.view(-1, 1, 1, 1)
    x0 = (sample.xt - (1 - alpha_bars_t).sqrt() * noise) / alpha_bars_t.sqrt()
    return torch.clamp((x0 + 1) / 2, 0, 1)

def plot_sanity_check(alpha_bars, sample, eps, device):
    x0 = original_image(alpha_bars, sample, eps).to(device)

    print("Original images")
    utils.plot_multiple_images(x0[:8])
    plt.show()
    print("Time steps:", sample.t[:8].view(-1).tolist())
    print("Noisy images")
    utils.plot_multiple_images(sample.xt[:8])
    plt.show()
    print("Noise to predict")
    utils.plot_multiple_images(eps[:8])
    plt.show()