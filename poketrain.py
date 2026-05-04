from torch.utils.data import DataLoader
from datasets.pokedataset import PokemonDataset
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

def process_data(dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return loader  

class Encoder(nn.Module):  
  def __init__(self, input_dim=49152, hidden_dim=1024, latent_dim=128):  
    super(Encoder, self).__init__()  
    self.fc1 = nn.Linear(input_dim, hidden_dim)  
    self.fc_mu = nn.Linear(hidden_dim, latent_dim)  
    self.fc_logvar = nn.Linear(hidden_dim, latent_dim) 

  def forward(self, x): 
    h = torch.relu(self.fc1(x)) 
    mu = self.fc_mu(h) 
    logvar = self.fc_logvar(h) 
    return mu, logvar 

class Decoder(nn.Module):  
  def __init__(self, input_dim=49152, hidden_dim=1024, latent_dim=128):  
    super(Decoder, self).__init__()  
    self.fc1 = nn.Linear(latent_dim, hidden_dim)  
    self.fc2 = nn.Linear(hidden_dim, output_dim)  

  def forward(self, z):  
    h = torch.relu(self.fc1(z))  
    return torch.sigmoid(self.fc2(h)) 

class VAE(nn.Module):  
  def __init__(self, input_dim=49152, hidden_dim=1024, latent_dim=128):  
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



def main():
    dataset = PokemonDataset("data/images")
    loader = process_data(dataset)
    images = next(iter(loader))
    plt.imshow(images[0].permute(1, 2, 0))
    plt.show()

if __name__ == "__main__":
    main()