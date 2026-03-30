from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------------
# Custom Collate functions
# -------------------------
def mtg_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate:
      - stack x_in / x_gt into tensors
      - keep creature_types as list-of-lists (no stacking)
      - keep image_path as list of strings
    """
    x_in = torch.stack([b["x_in"] for b in batch], dim=0)
    x_gt = torch.stack([b["x_gt"] for b in batch], dim=0)

    creature_types = [b["creature_types"] for b in batch]
    image_path = [b["image_path"] for b in batch]

    return {
        "x_in": x_in,
        "x_gt": x_gt,
        "creature_types": creature_types,
        "image_path": image_path,
    }

def collate_views(batch: list[dict]) -> dict:
    """
    Collate for embedding extraction mode where each item has:
      views: (4,3,H,W)
      base_idx: int
      creature_types: list[str]
      image_path: str

    Returns:
      x: (B*4,3,H,W)
      base_idx: (B*4,) indices of original card repeated 4x
      view_id: (B*4,) 0..3
      creature_types: list[list[str]] length B (original)
      image_path: list[str] length B (original)
    """
    views = torch.stack([b["views"] for b in batch], dim=0)  # (B,4,3,H,W)
    B = views.shape[0]
    x = views.view(B * 4, *views.shape[2:])                 # (B*4,3,H,W)

    base_idx = torch.tensor([b["base_idx"] for b in batch], dtype=torch.long)
    base_idx = base_idx.repeat_interleave(4)                # (B*4,)

    view_id = torch.arange(4, dtype=torch.long).repeat(B)   # (B*4,)

    return {
        "x": x,
        "base_idx": base_idx,
        "view_id": view_id,
        "creature_types": [b["creature_types"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }

# -------------------------
# Project root helper
# -------------------------
def get_project_root() -> Path:
    """
    Assumes this file lives in <project_root>/scripts/dataloader.py
    so project root is parents[1].
    """
    return Path(__file__).resolve().parents[1]


def resolve_from_project_root(p: str | Path) -> Path:
    """
    If p is relative, interpret it relative to project root,
    not relative to current working directory.
    """
    p = Path(p)
    if p.is_absolute():
        return p
    return (get_project_root() / p)


# -------------------------
# Config
# -------------------------
@dataclass
class DataConfig:
    # relative to project root by default
    index_path: str = "data/index.jsonl"
    img_size: int = 384
    batch_size: int = 8
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    drop_last: bool = True

    # If True: return x_in (degraded) and x_gt (clean)
    denoise: bool = False

    # Flip prob
    flip_p: float = 0.5

    # Debug limit
    limit: int = 0  # 0 = no limit


# -------------------------
# Transforms
# -------------------------
def build_clean_transform(img_size: int, flip_p: float = 0.5) -> A.Compose:
    """
    Applied to BOTH input and ground-truth (image + image2).

    Goal:
      - Kill color/lighting shortcuts by strong photometric augmentation
      - Keep geometry mostly intact (so recon still makes sense)
    """
    # Strong-ish color jitter *but not so strong that GT becomes implausible*
    photometric = A.OneOf(
        [
            # broad color/brightness/contrast in one go
            A.ColorJitter(
                brightness=0.35,   # heavy
                contrast=0.35,     # heavy
                saturation=0.30,   # heavy
                hue=0.08,          # moderate; too high can look unreal
                p=1.0
            ),
            # gamma changes mimic exposure / scanning variation
            A.RandomGamma(gamma_limit=(70, 140), p=1.0),  # heavy-ish
            # tone curve can change overall rendering significantly
            A.RandomToneCurve(scale=0.30, p=1.0),
            # slightly shift RGB channels (like print/scan bias)
            A.RGBShift(r_shift_limit=18, g_shift_limit=18, b_shift_limit=18, p=1.0),
        ],
        p=0.85,
    )

    # Sometimes remove color info entirely to force structure usage
    # Keep probability not too high, otherwise everything becomes grayscale-ish.
    color_removal = A.OneOf(
        [
            A.ToGray(p=1.0),
            A.ChannelShuffle(p=1.0),  # breaks "red=fire" shortcuts etc.
        ],
        p=0.15,
    )

    # Mild blur/noise on BOTH is okay occasionally; helps robustness without harming target too much
    mild_shared_corrupt = A.OneOf(
        [
            A.GaussNoise(std_range=(0.0, 0.010), mean_range=(0.0, 0.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 0.4), p=1.0),
        ],
        p=0.10,
    )

    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),

            # hmm
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.03, rotate_limit=3,border_mode=cv2.BORDER_REFLECT_101, p=0.05),

            A.HorizontalFlip(p=flip_p),

            photometric,
            color_removal,
            mild_shared_corrupt,

            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
        additional_targets={"image2": "image"},
    )


def build_degrade_transform() -> A.Compose:
    """
      - Primary always applied (p=1.0)
      - Local corruption very common
      - Photometric frequently
      - Motion blur (important for faces / structure)
    """

    # Primary degradation: ALWAYS apply one
    primary = A.OneOf(
        [
            A.GaussianBlur(blur_limit=(3, 11), sigma_limit=(0.3, 2.2), p=1.0),

            # ↓ slightly harsher downscale (forces reconstruction of detail)
            A.Downscale(scale_range=(0.60, 0.92), interpolation=cv2.INTER_AREA, p=1.0),

            # ↓ allow worse compression artifacts
            A.ImageCompression(quality_range=(35, 85), p=1.0),

            # ↓ slightly stronger noise
            A.GaussNoise(std_range=(0.005, 0.030), mean_range=(0.0, 0.0), p=1.0),

            # NEW: motion blur → very important for structure learning
            A.MotionBlur(blur_limit=7, p=1.0),
        ],
        p=1.0,  # was 0.90
    )

    # Secondary local corruption (more frequent)
    local = A.OneOf(
        [
            A.CoarseDropout(
                num_holes_range=(2, 24),
                hole_height_range=(10, 36),
                hole_width_range=(10, 36),
                fill=0,
                p=1.0,
            ),
            A.PixelDropout(dropout_prob=0.03, p=1.0),
        ],
        p=0.30,  # was 0.20
    )

    # Photometric corruption ONLY on input (slightly more frequent)
    photometric_only_input = A.OneOf(
        [
            A.ColorJitter(
                brightness=0.50,
                contrast=0.50,
                saturation=0.45,
                hue=0.12,
                p=1.0,
            ),
            A.RandomGamma(gamma_limit=(55, 170), p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.40,
                contrast_limit=0.40,
                p=1.0
            ),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0),
        ],
        p=0.80,  # was 0.70
    )

    # Slightly more frequent grayscale (forces structure reliance)
    color_drop_input = A.ToGray(p=0.18)  # was 0.12

    # Slightly more sharpening artifacts (teaches inverse)
    sharpen = A.Sharpen(alpha=(0.05, 0.35), lightness=(0.85, 1.15), p=0.15)

    return A.Compose(
        [
            primary,
            local,
            photometric_only_input,
            color_drop_input,
            sharpen,
        ]
    )

def build_embed_transform(img_size: int) -> A.Compose:
    """
    Deterministic transform for embedding extraction:
    - no flip
    - no jitter
    - no noise
    """
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
        A.ToFloat(max_value=255.0),
        # NOTE: No ToTensorV2() here because still 4 views in numpy needed
    ])


# -------------------------
# Dataset (JSONL only)
# -------------------------
class MTGCreatureArtDataset(Dataset):
    """
    Expects a JSONL file at index_path:
      {"image_path": "...", "creature_types": [...], ...}
      {"image_path": "...", "creature_types": [...], ...}
    """

    def __init__(
        self,
        index_path: str | Path,
        img_size: int = 384,
        flip_p: float = 0.5,
        denoise: bool = False,
        seed: int = 123,
        limit: int = 0,
        return_views: bool = False,
    ):
        self.project_root = get_project_root()
        self.index_path = resolve_from_project_root(index_path).resolve()
        self.return_views = return_views

        if self.index_path.suffix.lower() != ".jsonl":
            raise ValueError(f"Expected .jsonl index file, got: {self.index_path}")

        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        # used for relative image paths
        self.index_dir = self.index_path.parent

        self.img_size = img_size
        self.denoise = denoise
        self.rng = random.Random(seed)

        # Load JSONL entries
        entries: List[Dict[str, Any]] = []
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    if isinstance(e, dict):
                        entries.append(e)
                except json.JSONDecodeError:
                    continue

        if not entries:
            raise RuntimeError(f"No valid entries found in: {self.index_path}")

        if limit and limit > 0:
            entries = entries[:limit]

        self.entries = entries

        # Transforms
        self.tf_clean = build_clean_transform(img_size, flip_p)
        self.tf_degrade = build_degrade_transform() if denoise else None

    def __len__(self) -> int:
        return len(self.entries)

    def _resolve_image_path(self, rel_or_abs: str) -> Path:
        """
        Robust path resolution for image_path stored in JSONL:

        1) as-is
        2) relative to index folder (data/)
        3) relative to project root
        4) POSIX-split fallback relative to index folder
        """
        p = Path(rel_or_abs)

        if p.exists():
            return p.resolve()

        p2 = (self.index_dir / rel_or_abs)
        if p2.exists():
            return p2.resolve()

        p3 = (self.project_root / rel_or_abs)
        if p3.exists():
            return p3.resolve()

        p4 = self.index_dir.joinpath(*rel_or_abs.split("/"))
        if p4.exists():
            return p4.resolve()

        raise FileNotFoundError(f"Image not found: {rel_or_abs}")

    @staticmethod
    def _read_image_bgr(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return img

    @staticmethod
    def _square_crop_min_edge(img: np.ndarray, rng: random.Random) -> np.ndarray:
        """
        Random square crop of size min(H,W) from the original image.
        """
        h, w = img.shape[:2]
        s = min(h, w)
        if h == s and w == s:
            return img

        y0 = 0 if h == s else rng.randint(0, h - s)
        x0 = 0 if w == s else rng.randint(0, w - s)
        return img[y0:y0 + s, x0:x0 + s]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        e = self.entries[idx]
        rel_path = e.get("image_path") or e.get("path")
        if not rel_path:
            raise KeyError("Entry missing 'image_path'/'path'.")

        img_path = self._resolve_image_path(rel_path)
        img_bgr = self._read_image_bgr(img_path)

        # random square crop first (kept; if you want fully deterministic embeddings,
        # change this to center-crop or seeded by idx)
        img_bgr = self._square_crop_min_edge(img_bgr, self.rng)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        creature_types = e.get("creature_types", [])

        # ------------------------------------------------------------
        # Embedding mode — return 4 deterministic views
        # ------------------------------------------------------------
        if getattr(self, "return_views", False):
            # Resize + float once
            out = self.tf_embed(image=img_rgb)["image"]  # HWC float32 in [0,1]
            # Ensure float32
            if out.dtype != np.float32:
                out = out.astype(np.float32)

            # Build grayscale (still HWC, 3ch)
            gray = cv2.cvtColor((out * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray = (gray.astype(np.float32) / 255.0)
            gray3 = np.stack([gray, gray, gray], axis=-1)  # HWC

            # Flip (horizontal)
            out_flip = out[:, ::-1, :]
            gray3_flip = gray3[:, ::-1, :]

            # Convert 4 views -> torch (4,3,H,W)
            def hwc_to_chw(x: np.ndarray) -> torch.Tensor:
                return torch.from_numpy(x).permute(2, 0, 1).contiguous()

            v0 = hwc_to_chw(out)
            v1 = hwc_to_chw(out_flip)
            v2 = hwc_to_chw(gray3)
            v3 = hwc_to_chw(gray3_flip)

            views = torch.stack([v0, v1, v2, v3], dim=0)  # (4,3,H,W)

            return {
                "views": views,                   # NEW key
                "base_idx": idx,                  # to aggregate back per-card
                "creature_types": creature_types,
                "image_path": str(img_path),
            }

        # ------------------------------------------------------------
        # Training AE
        # ------------------------------------------------------------
        e = self.entries[idx]
        rel_path = e.get("image_path") or e.get("path")
        if not rel_path:
            raise KeyError("Entry missing 'image_path'/'path'.")

        img_path = self._resolve_image_path(rel_path)
        img_bgr = self._read_image_bgr(img_path)

        # random square crop first
        img_bgr = self._square_crop_min_edge(img_bgr, self.rng)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # If denoise=True, degrade first in numpy space, then apply clean TF to both.
        if not self.denoise:
            out = self.tf_clean(image=img_rgb, image2=img_rgb)
            x_in = out["image"]
            x_gt = out["image2"]
        else:
            img_deg = self.tf_degrade(image=img_rgb)["image"]
            out = self.tf_clean(image=img_deg, image2=img_rgb)
            x_in = out["image"]
            x_gt = out["image2"]

        return {
            "x_in": x_in,
            "x_gt": x_gt,
            "creature_types": e.get("creature_types", []),
            "image_path": str(img_path),
        }

# -------------------------
# Dataloader helper
# -------------------------
def make_dataloader(cfg: DataConfig, seed: int = 123) -> DataLoader:
    ds = MTGCreatureArtDataset(
        index_path=cfg.index_path,
        img_size=cfg.img_size,
        flip_p=cfg.flip_p,
        denoise=cfg.denoise,
        seed=seed,
        limit=cfg.limit,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        collate_fn=mtg_collate,
    )



# -------------------------
# Debug
# -------------------------
def _stats(x: torch.Tensor) -> str:
    return (
        f"shape={tuple(x.shape)} dtype={x.dtype} "
        f"min={x.min().item():.6f} max={x.max().item():.6f} "
        f"mean={x.mean().item():.6f} std={x.std().item():.6f} "
        f"finite={torch.isfinite(x).all().item()}"
    )


if __name__ == "__main__":
    cfg = DataConfig(
        index_path="data/index.jsonl",
        img_size=384,
        batch_size=8,
        num_workers=0,
        denoise=True,  # toggle
        limit=0,
    )

    print("=== dataloader.py debug ===")
    print("CWD         :", os.getcwd())
    print("Project root:", get_project_root())
    print("Index (cfg)  :", cfg.index_path)
    print("Index (abs)  :", resolve_from_project_root(cfg.index_path).resolve())
    print("Exists       :", resolve_from_project_root(cfg.index_path).exists())
    print("===========================")

    dl = make_dataloader(cfg)
    batch = next(iter(dl))
    print("x_in:", _stats(batch["x_in"]))
    print("x_gt:", _stats(batch["x_gt"]))
    print("types:", batch["creature_types"][0])
    print("path :", batch["image_path"][0])
