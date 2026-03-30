# 🧠 MTG Latent Autoencoder

Learn and explore a structured latent space for Magic: The Gathering creature artwork using a denoising autoencoder, embedding directions, and interactive latent manipulation.

This project focuses on:

* reconstructing and denoising artwork
* learning meaningful latent representations
* enabling controlled semantic edits (e.g. creature type transformations)

---

# 📸 Examples

<!-- Add your images here -->

<!--
![example1](path/to/image1.png)
![example2](path/to/image2.png)
-->

---

# ⚙️ Requirements

### Core

* Python 3.10+
* CUDA-capable GPU (recommended)

### Python packages

```bash
pip install torch torchvision
pip install albumentations opencv-python
pip install numpy pillow tqdm
```

### Notes

* PyTorch should match your CUDA version
  → https://pytorch.org/get-started/locally/
* Works on CPU, but **training will be very slow**

---

# 🚀 Usage

Follow these steps in order:

---

## 1. Train the autoencoder

```bash
python scripts/train_autoencoder.py
```

Trains the base model for reconstruction and denoising.

---

## 2. Train the patch critic (optional but recommended)

```bash
python scripts/create_train_patch_critic.py
```

Adds a learned perceptual signal:

* global structure
* local detail
* artistic quality

---

## 3. Create latent embeddings

```bash
python scripts/save_embeddings.py
```

Computes per-creature-type latent directions:

```
delta = mean(type) - global_mean
```

These are later used for semantic manipulation.

---

## 4. Explore latent space interactively

```bash
python latent_manipulator.py
```

Features:

* random dataset sampling
* latent sliders for creature types
* iterative refinement (multi-pass decoding)
* image upload support

---

# 🧪 Notes

* Latent manipulation works best with **normalized embedding directions**
* Strong augmentations improve robustness but can reduce sharpness
* Iterative decoding (feeding output back into the model) can enhance details

---

# 📌 Todo / Ideas

* Better critic convergence
* Orthogonalized latent directions
* Improved face/detail reconstruction
* Diffusion-style refinement

---

# 📄 License

MIT (or your preferred license)

