# scripts/view_reconstruction_pygame.py
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pygame
import torch
from PIL import Image


# -------------------------
# Utilities
# -------------------------
def project_root() -> Path:
    # scripts/view_reconstruction_pygame.py -> scripts -> project root
    return Path(__file__).resolve().parents[1]


def ensure_scripts_on_path():
    scripts_dir = project_root() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def tensor_to_pil_rgb(x: torch.Tensor) -> Image.Image:
    """
    x: (3,H,W) float32 in [0,1]
    """
    x = x.detach().clamp(0, 1).cpu()
    x = (x * 255.0).round().byte()
    arr = x.permute(1, 2, 0).numpy()  # HWC
    return Image.fromarray(arr, mode="RGB")


def pil_to_surface(img: Image.Image) -> pygame.Surface:
    img = img.convert("RGB")
    data = img.tobytes()
    return pygame.image.fromstring(data, img.size, "RGB")


def fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = img.size
    scale = min(max_w / w, max_h / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def draw_text(screen: pygame.Surface, font: pygame.font.Font, text: str, x: int, y: int, color=(220, 220, 220)):
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))


def take_recon(model_out: torch.Tensor | tuple | list | dict) -> torch.Tensor:
    """
    Normalize model outputs to the reconstruction image tensor (B,3,H,W).

    Supports:
      - y
      - (y, logits)
      - (y, z, logits)
      - {"y": y, ...}  (if you ever change to dict)
    """
    if isinstance(model_out, (tuple, list)):
        if len(model_out) >= 1:
            return model_out[0]
        raise RuntimeError("Model returned an empty tuple/list.")
    if isinstance(model_out, dict):
        if "y" in model_out:
            return model_out["y"]
        # fallback: try common key
        for k in ("x_hat", "recon", "out"):
            if k in model_out:
                return model_out[k]
        raise RuntimeError(f"Model returned dict but no recon key found. Keys={list(model_out.keys())}")
    return model_out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Browse AE reconstructions side-by-side (pygame).")
    ap.add_argument("--index", type=str, default="data/index.jsonl", help="Path to index.jsonl (relative to project root).")
    ap.add_argument("--model", type=str, default="models/model.pt", help="Path to AE checkpoint .pt (relative to project root).")
    ap.add_argument("--out_size", type=int, default=384, help="Model input size (must match the model).")
    ap.add_argument("--window_w", type=int, default=1200)
    ap.add_argument("--window_h", type=int, default=680)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    ensure_scripts_on_path()

    # Import your project modules (must exist in scripts/)
    from dataloader import MTGCreatureArtDataset
    from create_autoencoder_model import AEConfig, Autoencoder

    root = project_root()
    index_path = (root / args.index).resolve()
    model_path = (root / args.model).resolve()

    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Load checkpoint
    payload = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "config" in payload and "state_dict" in payload:
        cfg_dict = payload["config"]
        state_dict = payload["state_dict"]
    else:
        raise RuntimeError("Checkpoint format unexpected. Expected dict with keys: config, state_dict.")

    cfg = AEConfig(**cfg_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder(cfg).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Loaded model:", model_path)
    print("Config:", json.dumps(cfg_dict, indent=2))
    print("Device:", device)

    # Dataset (use clean input; no denoise; deterministic flip)
    ds = MTGCreatureArtDataset(
        index_path=index_path,
        img_size=cfg.img_size,
        flip_p=0.0,
        denoise=False,
        seed=args.seed,
        limit=0,
    )
    rng = random.Random(args.seed)

    # Pygame setup
    pygame.init()
    try:
        W, H = args.window_w, args.window_h
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("MTG AE Reconstruction Viewer (SPACE next, ESC quit)")
        clock = pygame.time.Clock()

        font = pygame.font.SysFont(None, 26)
        small = pygame.font.SysFont(None, 20)

        # Layout
        pad = 14
        top_bar = 56
        hint_bar = 28
        content_h = H - top_bar - hint_bar - pad
        panel_w = (W - pad * 3) // 2
        panel_h = content_h

        left_rect = pygame.Rect(pad, top_bar, panel_w, panel_h)
        right_rect = pygame.Rect(pad * 2 + panel_w, top_bar, panel_w, panel_h)

        current = {
            "surf_in": None,
            "surf_out": None,
            "title": "",
            "types": "",
            "path": "",
        }

        def pick_new():
            idx = rng.randrange(len(ds))
            sample = ds[idx]
            x_in = sample["x_in"]  # (3,H,W) float in [0,1]
            creature_types = sample.get("creature_types", [])
            img_path = sample.get("image_path", "")

            # Forward pass (ignore classifier head etc.)
            with torch.no_grad():
                xin_b = x_in.unsqueeze(0).to(device)
                out = model(xin_b)
                y_b = take_recon(out)
                y = y_b.squeeze(0).detach().clamp(0, 1)

            pil_in = tensor_to_pil_rgb(x_in)
            pil_out = tensor_to_pil_rgb(y)

            pil_in_fit = fit_image(pil_in, left_rect.w, left_rect.h)
            pil_out_fit = fit_image(pil_out, right_rect.w, right_rect.h)

            current["surf_in"] = pil_to_surface(pil_in_fit)
            current["surf_out"] = pil_to_surface(pil_out_fit)
            current["title"] = Path(img_path).name if img_path else f"idx={idx}"
            current["types"] = ", ".join(creature_types) if creature_types else "(no types)"
            current["path"] = img_path

        pick_new()

        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_SPACE:
                        pick_new()

            screen.fill((10, 10, 10))

            draw_text(screen, font, "Input (model) vs Reconstruction", pad, 14)
            draw_text(screen, small, f"File: {current['title']}", pad, 34, color=(170, 170, 170))

            pygame.draw.rect(screen, (18, 18, 18), left_rect, border_radius=8)
            pygame.draw.rect(screen, (18, 18, 18), right_rect, border_radius=8)

            if current["surf_in"] is not None:
                s = current["surf_in"]
                x = left_rect.x + (left_rect.w - s.get_width()) // 2
                y = left_rect.y + (left_rect.h - s.get_height()) // 2
                screen.blit(s, (x, y))

            if current["surf_out"] is not None:
                s = current["surf_out"]
                x = right_rect.x + (right_rect.w - s.get_width()) // 2
                y = right_rect.y + (right_rect.h - s.get_height()) // 2
                screen.blit(s, (x, y))

            info_y = H - hint_bar
            pygame.draw.rect(screen, (15, 15, 15), pygame.Rect(0, info_y, W, hint_bar))
            draw_text(screen, small, f"Types: {current['types']}", pad, info_y + 5, color=(200, 200, 200))

            hint = "SPACE = next random    |    ESC = quit"
            hint_surf = small.render(hint, True, (150, 150, 150))
            screen.blit(hint_surf, (W - hint_surf.get_width() - pad, info_y + 5))

            pygame.display.flip()
            clock.tick(60)

    finally:
        pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)