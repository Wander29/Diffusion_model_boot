import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from abc import abstractmethod
from typing import override, List, Any
from dataclasses import dataclass

from vae_ludo.types import Tensor

@dataclass
class VAEOutput:
    """
    Dataclass for VAE output.
    
    Attributes:
        x_recon (Tensor): The reconstructed output from the VAE
        y_pred (Tensor): (optional) The predicted label from the optional classifier
        z (Tensor): The sampled value of the latent variable z
        mu (Tensor): Mean of latent Gaussian from the output of the encoder
        log_var (Tensor): Log variance of latent Gaussian from the output of the encoder
        std_dev (Tensor): Std deviation of latent Gaussian from the output of the encoder
    """
    x_recon: Tensor
    z: Tensor
    mu: Tensor
    log_var: Tensor
    std_dev: Tensor
    y_pred: Tensor=torch.Tensor([])
    
class BaseVAE(nn.Module):
    """
    @Kingma, D. P. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
  
    The full network structure looks like:
        Encoder:
            Input > [hidden layers] > Mean & Log-Variance outputs

        Decoder:
            Latent z > [hidden layers] > Output reconstruction

    Encoder: 
        - Input: @input_dim is the size of the input vector
        - Output: Two vectors of size @latent_dim (for mean and log-variance)

    Decoder:
        - Input: size = @latent_dim, it's the size of latent vector z
        - Output: size = @input_dim, it's the size of reconstructed output (same as encoder input)
    """

    def __init__(self, input_dim, latent_dim) -> None:
      super(BaseVAE, self).__init__()
      self.input_dim =  input_dim
      self.latent_dim = latent_dim

    @abstractmethod
    def encode(self, x: Tensor) -> List[Tensor]:
      pass

    def decode(self, z: Tensor) -> Any:
      """
        Decodes data from latent space to the original input space

        Args:
          @z (Tensor)    Data in latent space

        Returns: (Tensor) reconstructed data in the original input space
      """
      return self.decoder(z)

    # def sample(self, batch_size:int) -> Tensor:
    #   raise NotImplementedError

    # def generate(self, x: Tensor) -> Tensor:
    #   raise NotImplementedError

    @abstractmethod
    def forward(self, x: Tensor) -> VAEOutput:
      pass

class GaussianVAE(BaseVAE):
  
  def __init__(self, input_dim, latent_dim):
    super(GaussianVAE, self).__init__(input_dim, latent_dim)

  def reparameterize(self, mu: Tensor, std_dev: Tensor) -> Tensor:
    """
    Reparameterization trick to obtain a sample from N(mu, var) by 
    sampling only from N(0,1).
    
    Args:
      @mu (Tensor) Mean of the Gaussian
      @std_dev (Tensor) Standard Deviation of the Gaussian
    
    Returns: (Tensor)
    """

    eps = torch.randn_like(std_dev)
    return eps * std_dev + mu

  @override
  def encode(self, x: Tensor) -> List[Tensor]:
    """
      Encodes the input data into the latent space by passing through the encoder network

      Args:
        @x      : Tensor     input data
    """
    # @x has shape [batch_size, input_dim]
    # @out has shape [batch_size, latent_dim*2]
    out = self.encoder(x)
    # chunk: Attempts to split a tensor into the specified number of chunks
    mu, log_var = torch.chunk(out, 2, dim=-1)
    
    ### Clamp values outside acceptable range
    # NOTE: 
    # - This clamping affects gradient
    # mu = torch.clamp(mu, min=-20.0, max=20.0)
    # log_var = torch.clamp(log_var, min=-20.0, max=20.0)
    #
    # - use this strategy to clamp without affecting gradient
    mu = mu + (torch.clamp(mu, -40.0, 40.0) - mu).detach()
    log_var = log_var + (torch.clamp(log_var, -40.0, 40.0) - log_var).detach()

    std_dev = torch.sqrt(torch.exp(log_var)) # compute it one time
    return [mu, log_var, std_dev]
  
  @staticmethod
  def encoded_distribution(mu: Tensor, std_dev:Tensor, eps:float=1.e-5) -> MultivariateNormal:
    """
    Args:
      @mu         (Tensor)  mean(s) of the latent normal
      @std_dev    (Tensor)  std deviation(s) of the latent normal
      @eps        (float)   epsilon to improve numerical stability (we need positive definite covariance matrices)
    """
    # `scale_tril` is the Cholesky factor of the covariance matrix, 
    # which is based on the std deviation, since 
    # `scale_tril @ scale_tril.T = covariance_matrix`
    scale_tril = torch.diag_embed(std_dev+eps)
    latent_normal = MultivariateNormal(mu, scale_tril=scale_tril)
    return latent_normal

  def latent_encoding(self, x: Tensor) -> Tensor:
    mu, _, std_dev = self.encode(x)
    z = self.reparameterize(mu, std_dev=std_dev)
    
    return z
  
  @override
  def forward(self, x: Tensor) -> VAEOutput:
    """
    Args:
      @x (Tensor) input data
    
    Returns: (VAEOutput)
    """
    # x = x.view(x.size(0), -1) # flatten input
    mu, log_var, std_dev = self.encode(x)
    z = self.reparameterize(mu, std_dev=std_dev)

    return VAEOutput(
      x_recon = self.decode(z),
      z = z, 
      mu = mu, 
      log_var = log_var,
      std_dev = std_dev
    )