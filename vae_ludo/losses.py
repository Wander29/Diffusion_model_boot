import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from vae_ludo.types import Tensor
from vae_ludo.base_vae import VAEOutput, GaussianVAE

class VAELoss(nn.Module):
  def __init__(self, lkl):
    super(VAELoss, self).__init__()
    self.lkl = lkl                                    # lambda for KL divergence

  @staticmethod
  def reconstruction_error(x: Tensor, x_recon:Tensor) -> Tensor:
    recon_loss = torch.mean(F.mse_loss(x_recon, x, reduction = 'none'), dim=1)
    return recon_loss

  @staticmethod
  def kl_divergence(out: VAEOutput):
    latent_normal = GaussianVAE.encoded_distribution(mu=out.mu, std_dev=out.std_dev)
    std_normal = MultivariateNormal(torch.zeros_like(out.z), scale_tril=torch.eye(out.z.shape[-1]))
    kld_loss = torch.distributions.kl.kl_divergence(latent_normal, std_normal) 

    return kld_loss

  def forward(self, out: VAEOutput, targets):
    """
    Computes the VAE loss function for a deterministic decoder:
      -KL_D (q(z)||p(z)) + Reconstruction error
     
    It's possible to get the single contributions from the loss
    
    Args:
        predictions (torch.Tensor):   Model predictions
        targets (torch.Tensor):       Ground truth targets
    """
    self.recon_loss = F.mse_loss(out.x_recon, targets, reduction = 'mean')
    self.kld_loss = VAELoss.kl_divergence(out).mean()

    self.loss = self.recon_loss + self.lkl * self.kld_loss
    
    return self.loss
    
  def __getitem__(self, key):
    """
    Allow accessing loss components or parameters
    
    Args:
        key (str): Key to retrieve loss component or parameter
    
    Returns:
        Value associated with the key
    """
    # Store some loss-related metrics or components
    if key == '':
      return self.loss
    elif key == 'kld':
        return self.kl_divergence
    elif key == 'reconstruction':
        return self.recon_loss
    else:
        raise KeyError(f"No loss component found for key: {key}")
        
