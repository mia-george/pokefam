import os
from torch.utils.data import DataLoader
from datasets.pokedataset import PokemonDataset
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse

def show_reconstructions(model, dataset, device, pokemon_names):
    name_to_idx = {os.path.splitext(f)[0].lower(): i for i, f in enumerate(dataset.image_files)}
    indices = []
    for name in pokemon_names:
        idx = name_to_idx.get(name.lower())
        if idx is None:
            raise ValueError(f"'{name}' not found in dataset.")
        indices.append(idx)
    n = len(indices)
    model.eval()
    with torch.no_grad():
        samples = torch.stack([dataset[i] for i in indices])  
        x = samples.to(device) 
        recon_x, _, _ = model(x)
        originals = samples.cpu()
        reconstructed = recon_x.cpu()

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        axes[0, i].imshow(originals[i].permute(1, 2, 0).clamp(0, 1))
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).clamp(0, 1))
        axes[1, i].axis('off')

    axes[0, 0].set_title("Original", fontsize=10)
    axes[1, 0].set_title("Reconstructed", fontsize=10)
    plt.tight_layout()
    plt.show()


def show_generated(model, device, n=8, latent_dim=256):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim).to(device)
        generated = model.decoder(z).view(n, 3, 64, 64).cpu()

    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    for i in range(n):
        axes[i].imshow(generated[i].permute(1, 2, 0).clamp(0, 1))
        axes[i].axis('off')

    plt.suptitle("VAE Generated Images")
    plt.tight_layout()
    plt.show()


class Encoder(nn.Module):  
  def __init__(self, latent_dim=256):  
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
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
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
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

def loss_function(recon_x, x, mu, logvar, epoch, warmup_epochs=50):
    batch_size = x.size(0)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    kl_weight = min(1.0, epoch / warmup_epochs)
    return recon_loss + kl_weight * kl_loss


def main():
    # Argument handling
    parser = argparse.ArgumentParser(description="Train a VAE on Pokémon images.")
    parser.add_argument(
        "pokemon",
        nargs=2,
        metavar="POKEMON",
        help="Enter 2 Pokémon parent names to generate an image of their child. The Pokémon number indicates which evolution to use. e.g. python poketrain.py pikachu1 charmander0"
    )
    args = parser.parse_args()

    # Load dataset
    dataset = PokemonDataset("data/images") 
    print(f"Training on all {len(dataset)} images. Will visualize: {args.pokemon[0]}, {args.pokemon[1]}")
    pokeloader = DataLoader(
        dataset,
        batch_size=64,      
        shuffle=True,
        num_workers=2,      # parallel loading
        pin_memory=True     # faster CPU→GPU transfer
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 400
    learning_rate = 1e-3  

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
        loss = loss_function(recon_x, x, mu, logvar, epoch + 1, warmup_epochs=50)
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

      if (epoch + 1) % 10 == 0:
          print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Plotting training loss 
    plt.plot(train_losses) 
    plt.title("VAE Training Loss") 
    plt.xlabel("Epoch") 
    plt.ylabel("Loss") 
    plt.grid(True) 
    plt.show() 

    show_reconstructions(model, dataset, device, args.pokemon)
    show_generated(model, device)



if __name__ == "__main__":
    main()