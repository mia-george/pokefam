from torch.utils.data import DataLoader
from datasets.pokedataset import PokemonDataset
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F 

def show_reconstructions(model, dataset, device, n=8):
    model.eval()
    with torch.no_grad():
        samples = torch.stack([dataset[i] for i in range(n)])  # (n, 3, 64, 64)
        x = samples.view(n, -1).to(device)
        recon_x, _, _ = model(x)
        originals = samples.cpu()
        reconstructed = recon_x.view(n, 3, 64, 64).cpu()

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


def show_generated(model, device, n=8, latent_dim=128):
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
  def __init__(self, input_dim=12288, hidden_dim=512, latent_dim=128):  
    super(VAE, self).__init__()  
    self.encoder = Encoder(input_dim, hidden_dim, latent_dim)  
    self.decoder = Decoder(latent_dim, hidden_dim, input_dim)  
 
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
  # Reconstruction loss (binary cross entropy)  
  recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')  

  # KL divergence loss  
  kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  
  return recon_loss + kl_loss 


def main():
    dataset = PokemonDataset("data/images")
    pokeloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # images = next(iter(loader))
    # plt.imshow(images[0].permute(1, 2, 0))
    # plt.show()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 20
    learning_rate = 1e-3  

    # Initialize model, optimizer 
    model = VAE().to(device)  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    # Track losses 
    train_losses = [] 

    # Training loop
    model.train()  

    for epoch in range(epochs):  
      total_loss = 0  
      for batch_idx, x in enumerate(pokeloader):  
        x = x.view(-1, 12288).to(device)  # Flatten images 
        optimizer.zero_grad() 

        recon_x, mu, logvar = model(x) 
        loss = loss_function(recon_x, x, mu, logvar) 
        loss.backward() 
        optimizer.step() 
     
        total_loss += loss.item() 
 
      avg_loss = total_loss / len(pokeloader.dataset) 
      train_losses.append(avg_loss) 
      print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}") 

    # Plotting training loss 
    plt.plot(train_losses) 
    plt.title("VAE Training Loss") 
    plt.xlabel("Epoch") 
    plt.ylabel("Loss") 
    plt.grid(True) 
    plt.show() 

    show_reconstructions(model, dataset, device)
    show_generated(model, device)



if __name__ == "__main__":
    main()