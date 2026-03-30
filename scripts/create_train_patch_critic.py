from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dataloader import DataConfig, make_dataloader

# ---------------------------------------------------------------------
# Multi-Head Patch Critic
# ---------------------------------------------------------------------
class PatchCritic(nn.Module):
    def __init__(self, in_ch=3, base=64, n_down=4):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 4, 2, 1),
                nn.GroupNorm(min(16, cout), cout),
                nn.LeakyReLU(0.2, inplace=True),
            )

        ch = base
        layers = [
            nn.Conv2d(in_ch, ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        for _ in range(n_down - 1):
            layers.append(block(ch, ch * 2))
            ch *= 2

        layers += [
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        self.features = nn.Sequential(*layers)

        # 3 heads
        self.head_global = nn.Conv2d(ch, 1, 1)
        self.head_local = nn.Conv2d(ch, 1, 1)
        self.head_art = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        f = self.features(x)
        return {
            "global": self.head_global(f),
            "local": self.head_local(f),
            "art": self.head_art(f),
        }

# ---------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------
def build_tf_clean(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])

# -------------------------
# GLOBAL STRUCTURE (composition / geometry)
# -------------------------
def build_tf_global(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.OneOf([
            A.Perspective(scale=(0.08, 0.18), keep_size=True, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.35, p=1.0),
        ], p=1.0),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])

# -------------------------
# LOCAL DETAIL (faces, texture, sharpness)
# -------------------------
def build_tf_local(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.OneOf([
            A.GaussianBlur(blur_limit=(9, 21), sigma_limit=(1.5, 4.0), p=1.0),
            A.GaussNoise(std_range=(0.04, 0.12), p=1.0),
            A.CoarseDropout(
                num_holes_range=(8, 32),
                hole_height_range=(16, 64),
                hole_width_range=(16, 64),
                fill=0,
                p=1.0,
            ),
            A.PixelDropout(dropout_prob=0.05, p=1.0),
        ], p=1.0),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])

# -------------------------
# ARTISTIC / STYLE (card quality, rendering)
# -------------------------
def build_tf_art(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.OneOf([
            A.ImageCompression(quality_range=(5, 35), p=1.0),
            A.Posterize(num_bits=(2, 4), p=1.0),
            A.Compose([
                A.GaussNoise(std_range=(0.06, 0.18), p=1.0),
                A.MultiplicativeNoise(
                    multiplier=(0.5, 1.5),
                    per_channel=True,
                    elementwise=True,
                    p=1.0,
                ),
            ], p=1.0),
            A.Compose([
                A.GaussNoise(std_range=(0.04, 0.12), p=1.0),
                A.ImageCompression(quality_range=(10, 45), p=1.0),
                A.MotionBlur(blur_limit=(5, 11), p=1.0),
            ], p=1.0),
            A.Downscale(scale_range=(0.25, 0.6), p=1.0),
        ], p=1.0),
        A.OneOf([
            A.RGBShift(30, 30, 30, p=1.0),
            A.ChannelShuffle(p=1.0),
            A.HueSaturationValue(20, 30, 20, p=1.0),
        ], p=0.6),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def chw01_to_uint8(x):
    x = (x.clamp(0, 1) * 255).byte()
    return x.permute(1, 2, 0).cpu().numpy()

def apply_tf(x, tf):
    return tf(image=chw01_to_uint8(x))["image"]

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.jsonl")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save_path", default="models/patch_critic.pt")

    args = ap.parse_args()
    print("Arguments:", args)
    print("Index file exists:", Path(args.index).exists())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = PatchCritic().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    dl = make_dataloader(DataConfig(
        index_path=args.index,
        img_size=args.img_size,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        denoise=False,
    ))

    tf_clean = build_tf_clean(args.img_size)
    tf_global = build_tf_global(args.img_size)
    tf_local = build_tf_local(args.img_size)
    tf_art = build_tf_art(args.img_size)

    print("=== TRAIN MULTI-HEAD CRITIC ===")
    print("device:", device)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()

        loss_sum = 0.0

        for step, batch in enumerate(dl):
            x = batch["x_in"]

            xs, ys = [], []

            for i in range(x.size(0)):
                mode = np.random.choice(["clean", "global", "local", "art"])

                if mode == "clean":
                    xi = apply_tf(x[i], tf_clean)
                    y = 0.0
                elif mode == "global":
                    xi = apply_tf(x[i], tf_global)
                    y = 1.0
                elif mode == "local":
                    xi = apply_tf(x[i], tf_local)
                    y = 1.0
                else:
                    xi = apply_tf(x[i], tf_art)
                    y = 1.0

                xs.append(xi)
                ys.append(y)

            x_aug = torch.stack(xs).to(device)
            y = torch.tensor(ys, device=device)

            out = model(x_aug)

            losses = {}
            total_loss = 0.0

            for k in out:
                logits = out[k]
                target = y.view(-1, 1, 1, 1).expand_as(logits)
                l = bce(logits, target)
                losses[k] = l
                total_loss += l

            total_loss /= len(losses)

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            loss_sum += total_loss.item()

            # -------------------------
            # DEBUG PRINTS
            # -------------------------
            if step % 50 == 0:
                with torch.no_grad():
                    print(f"\nepoch {epoch:03d} step {step:04d} | loss {total_loss:.4f}")

                    for k in out:
                        probs = torch.sigmoid(out[k]).mean(dim=(2, 3)).squeeze(1)

                        clean_mask = (y == 0)
                        deg_mask = (y == 1)

                        m_clean = probs[clean_mask].mean().item() if clean_mask.any() else float("nan")
                        m_deg = probs[deg_mask].mean().item() if deg_mask.any() else float("nan")

                        print(f"  {k:6s}: clean={m_clean:.3f}  degraded={m_deg:.3f}")

        dt = time.time() - t0
        print(f"\nEpoch {epoch} done | avg loss {loss_sum/(step+1):.4f} | {dt:.1f}s")

        torch.save(model.state_dict(), args.save_path)
        print("saved ->", args.save_path)

if __name__ == "__main__":
    main()

