import os
from torch.utils.data import DataLoader
from datasets.pokedataset import PokemonDataset
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse


class Encoder(nn.Module):  
  def __init__(self, latent_dim=256):  
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
    )
    self.fc_mu = nn.Linear(256*4*4, latent_dim)
    self.fc_logvar = nn.Linear(256*4*4, latent_dim)

  def forward(self, x): 
    h = self.conv(x).view(x.size(0), -1)
    mu = self.fc_mu(h) 
    logvar = self.fc_logvar(h) 
    return mu, logvar 

class Decoder(nn.Module):  
  def __init__(self, latent_dim=256):  
    super().__init__()
    self.fc = nn.Linear(latent_dim, 256*4*4)
    self.deconv = nn.Sequential(
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid()
    )

  def forward(self, z):  
    h = self.fc(z).view(-1, 256, 4, 4)
    return self.deconv(h) 

class VAE(nn.Module):  
  def __init__(self, latent_dim=256):  
    super().__init__()  
    self.encoder = Encoder(latent_dim)  
    self.decoder = Decoder(latent_dim)  
 
  def reparameterize(self, mu, logvar):  
    std = torch.exp(0.5 * logvar)   
    eps = torch.randn_like(std)   
    return mu + eps * std                 

  def forward(self, x):  
    mu, logvar = self.encoder(x)  
    z = self.reparameterize(mu, logvar)  
    reconstructed = self.decoder(z)  
    return reconstructed, mu, logvar 

def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + 0.1 * kl_loss


def main():
    # Load dataset
    dataset = PokemonDataset("data/images") 
    print(f"Training on {len(dataset)} images. Parents: {args.pokemon[0]}, {args.pokemon[1]}")
    pokeloader = DataLoader(
        dataset,
        batch_size=64,      
        shuffle=True,
        num_workers=2,      # parallel loading
        pin_memory=True     # faster CPU→GPU transfer
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 200
    learning_rate = 5e-4

    # Early stopping
    patience_counter = 0
    early_stop_patience = 50  # stop if no improvement for specified number of epochs

    # Initialize model, optimizer 
    model = VAE().to(device)  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=10,
        factor=0.5,
        min_lr=1e-5
    )

    # Track losses 
    train_losses = [] 
    best_loss = float('inf')

    # Training loop
    model.train()  

    for epoch in range(epochs):  
      total_loss = 0  
      for batch_idx, x in enumerate(pokeloader):  
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad() 

        recon_x, mu, logvar = model(x)
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
     
        total_loss += loss.item() 
 
      avg_loss = total_loss / len(pokeloader.dataset) 
      train_losses.append(avg_loss) 
      scheduler.step(avg_loss)
      if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'vae_model.pth')
      else:
          patience_counter += 1
          if patience_counter >= early_stop_patience:
              print(f"Early stopping at epoch {epoch+1}")
              break

      if (epoch + 1) % 10 == 0:
          print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Plotting training loss 
    # plt.plot(train_losses) 
    # plt.title("VAE Training Loss") 
    # plt.xlabel("Epoch") 
    # plt.ylabel("Loss") 
    # plt.grid(True) 
    # plt.show() 

if __name__ == "__main__":
    main()