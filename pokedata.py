import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PokemonDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.image_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(".png")
        ]

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img