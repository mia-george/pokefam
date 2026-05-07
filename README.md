# Report

**Pokefam** generates "child" Pokemon by blending two parent Pokemon. It uses a Convolutional Variational Autoencoder (CVAE) to learn the features of different Pokemon and create unique hybrids.

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
