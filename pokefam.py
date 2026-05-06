import os
import argparse
from datasets.pokedataset import PokemonDataset
import torch
import matplotlib.pyplot as plt
from poketrain import VAE 

def find_pokemon(name_to_idx, name):
    idx = name_to_idx.get(name.lower())
    if idx is None:
        similar = [k for k in name_to_idx if k.startswith(name.lower().rstrip('0123456789'))]
        raise ValueError(f"'{name}' not found. Available: {similar}")
    return idx

def generate_hybrid(model, dataset, device, pokemon1, pokemon2, alpha=0.5):
    name_to_idx = {os.path.splitext(f)[0].lower(): i for i, f in enumerate(dataset.image_files)}

    idx1 = find_pokemon(name_to_idx, pokemon1)
    idx2 = find_pokemon(name_to_idx, pokemon2)

    img1 = dataset[idx1].unsqueeze(0).to(device)
    img2 = dataset[idx2].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        mu1, _ = model.encoder(img1)
        mu2, _ = model.encoder(img2)
        z = (1 - alpha) * mu1 + alpha * mu2
        hybrid = model.decoder(z)

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    axes[0].imshow(dataset[idx1].permute(1, 2, 0).clamp(0, 1))
    axes[0].set_title(pokemon1.capitalize())
    axes[0].axis('off')
    axes[1].imshow(hybrid.squeeze(0).cpu().permute(1, 2, 0).clamp(0, 1))
    axes[1].set_title("Child")
    axes[1].axis('off')
    axes[2].imshow(dataset[idx2].permute(1, 2, 0).clamp(0, 1))
    axes[2].set_title(pokemon2.capitalize())
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(f"hybrid_{pokemon1}_{pokemon2}.png")
    plt.show()
        

def main():
    # Argument handling
    parser = argparse.ArgumentParser(description="Train a VAE on Pokémon images.")
    parser.add_argument(
        "pokemon",
        nargs=2,
        metavar="POKEMON",
        help="Enter 2 Pokémon parent names to generate an image of their child. The Pokémon number indicates which version to use. e.g. python pokefam.py pikachu1 charmander0"
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = PokemonDataset("data/images") 

    # Load trained model
    model = VAE().to(device)
    model.load_state_dict(torch.load('vae_model.pth', map_location=device))

    # Generate hybrid pokemon
    generate_hybrid(model, dataset, device, args.pokemon[0], args.pokemon[1], alpha=0.5)


if __name__ == "__main__":
    main()