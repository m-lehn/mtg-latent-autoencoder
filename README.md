# 🧠 MTG Latent Autoencoder


<img width="1393" height="774" alt="image" src="https://github.com/user-attachments/assets/f45a1dad-836a-4d0f-b1eb-4b5b5aa85a38" />

<img width="1390" height="773" alt="image" src="https://github.com/user-attachments/assets/eb10564a-1ac8-45d6-ba66-fdc88a834374" />

<img width="1396" height="784" alt="Screenshot 2026-03-30 223049" src="https://github.com/user-attachments/assets/6178280b-42c4-40fb-98b3-3e1f7e30d717" />

Learn and explore a structured latent space for Magic: The Gathering creature artwork using a denoising autoencoder, embedding directions, and interactive latent manipulation.

This project focuses on:

* reconstructing and denoising artwork
* learning meaningful latent representations
* enabling controlled semantic edits (e.g. creature type transformations)

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
* Works on CPU, but **training will be very slow** (do not do this)

---

# 🚀 Usage

Follow these steps in order:

---

## 1. Get the Images

```bash
python scripts/scryfall_api_test.py
python scripts/download_scryfall_creature_art.py
python scripts/browse_dataset_pygame.py
```

Check out the API of scryfall and download the images.

## 2. Create and Train the patch critic (optional but recommended)

```bash
python scripts/create_train_patch_critic.py
```

Adds a learned perceptual signal:

* global structure
* local detail
* artistic quality

---

## 3. Create Models

```bash
python scripts/create_autoencoder_model.py
python scripts/train_autoencoder.py
```

Trains the base model for reconstruction and denoising.
Several trainings are needed, the right hyperparameters and arguments are important.
A simple start without data augmentation (denoising opion) or any additinal loss is recommended.

---

## 4. Create latent Embeddings

```bash
python scripts/save_embeddings.py
python scripts/save_embeddings_big.py
```

Computes per-creature-type latent directions:

$$delta = mean(type) - mean(global)$$

These are later used for semantic manipulation.
Choose only one option, the big option is trying to get more shape based information, instead of mostly color.

---

## 5. Explore latent space interactively

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

Still some work to do and many types of creatures aren't very common.

