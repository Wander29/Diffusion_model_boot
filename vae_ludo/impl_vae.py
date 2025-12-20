import torch.nn as nn

from vae_ludo import base_vae as base

class ConvVAE(base.GaussianVAE):
  def __init__(self, input_dim, latent_dim, dropout_rate):
    base.GaussianVAE.__init__(self, input_dim, latent_dim)
    
    self.encoder = nn.Sequential(
      # input: 1 x 28 x 28 
      nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1), # 8 x 14 x 14
      nn.BatchNorm2d(8),
      nn.LeakyReLU(),
      nn.Dropout2d(dropout_rate),
      
      nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1), # 16 x 7 x 7
      nn.BatchNorm2d(16),
      nn.LeakyReLU(),
      nn.Dropout2d(dropout_rate),
      
      nn.Flatten(),
      nn.Linear(16 * 7 * 7, 64), # 64
      nn.BatchNorm1d(64),
      nn.LeakyReLU(),
      nn.Dropout(dropout_rate),
      
      nn.Linear(64, latent_dim*2)  # mean and log variance
    )
    
    self.decoder = nn.Sequential(
      nn.Linear(latent_dim, 64),
      nn.BatchNorm1d(64),
      nn.LeakyReLU(),
      nn.Dropout(dropout_rate),
      
      nn.Linear(64, 7 * 7 * 16), # 16 x 7 x 7
      nn.BatchNorm1d(7 * 7 * 16),
      nn.LeakyReLU(),
      nn.Dropout(dropout_rate),
      nn.Unflatten(1, (16, 7, 7)),
      
      nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1), # 16 x 14 x 14
      nn.BatchNorm2d(16),
      nn.LeakyReLU(),
      nn.Dropout2d(dropout_rate),
      
      nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1), # 8 x 28 x 28
      nn.BatchNorm2d(8),
      nn.LeakyReLU(),
      nn.Dropout2d(dropout_rate),
      
      nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1) # 1 x 28 x 28
    )        

# self.encoder = nn.Sequential(
    #   nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
    #   nn.LeakyReLU(),
    #   nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
    #   nn.LeakyReLU(),
    #   nn.Flatten(),
    #   nn.Linear(64 * 7 * 7, latent_dim*2)  # mean and log variance
    # )
    
    # Decoder
    # self.decoder = nn.Sequential(
    #   nn.Linear(latent_dim, 7 * 7 * 32),
    #   nn.LeakyReLU(),
    #   nn.Unflatten(1, (32, 7, 7)),
    #   nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1),
    #   nn.LeakyReLU(),
    #   nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
    #   nn.LeakyReLU(),
    #   nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)
    # )