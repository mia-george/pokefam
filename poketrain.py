from torch.utils.data import DataLoader
from datasets.pokedataset import PokemonDataset

dataset = PokemonDataset("data/images")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# sanity check
import matplotlib.pyplot as plt

images = next(iter(loader))
plt.imshow(images[0].permute(1, 2, 0))
plt.show()