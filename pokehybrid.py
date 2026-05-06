import os
import argparse
from datasets.pokedataset import PokemonDataset
import torch
import matplotlib.pyplot as plt
from poketrain import VAE 

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = PokemonDataset("data/images") 

    # Load trained model
    model = VAE().to(device)
    model.load_state_dict(torch.load('vae_model.pth', map_location=device))


if __name__ == "__main__":
    main()