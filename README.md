# Report

### Project Overview
Pokéfam allows users to select two “parent” Pokémon, and it will generate a hybrid “child” Pokémon that contains features of the parents.
The dataset I’m using contains multiple versions of each Pokémon, so my project takes in user arguments that specify which Pokémon version to use. I’m using a convolutional VAE and training on a set of around 2,500 images. The encoder compresses the images into latent space, performs interpolation to calculate a blend between the latent representations of the parents, and then the decoder reconstructs this data into a new “child” Pokémon image.

Here are a few sample images (These are also some in the `samples` folder):

![alt text](samples/hybrid_pikachu1_charmander0.png)

![alt text](samples/hybrid_pikachu1_charmander3.png)

![alt text](samples/hybrid_alomomola2_electrode0.png)

As seen above, the child images are not great. VAEs are usually blurry and do not produce the best quality, but I can make out features of both Pokémon in the hybrid image. It’s more like the essence of the Pokemon than a true hybrid generation, but I was pleased to see that I could at least make out shapes and colors from both parents. I do think it worked better on Pokemon that had similar shapes.


# How to Run

### 1. Install Dependencies
Make sure you have the required libraries installed:
```bash
pip install -r requirements.txt
```

### 2. Generate a Hybrid
Run the `pokefam.py` script with the names of two parent Pokemon. You can find available names in `docs/pokemon_list.md` or the `data/images` folder (use the filename without the extension).

```bash
python pokefam.py <parent1> <parent2>
```

**Example:**
```bash
python pokefam.py pikachu0 charmander1
```

The resulting image will be saved in the root directory as `hybrid_<parent1>_<parent2>.png`.

### 3. Training (Optional)
If you want to train the model yourself, you can run:
```bash
python poketrain.py
```
*A pre-trained model (`vae_model.pth`) is already included.*
