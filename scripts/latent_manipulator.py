from __future__ import annotations

import argparse
import base64
import json
import random
import sys
import tkinter as tk
import numpy as np
from tkinter import filedialog
from pathlib import Path
from typing import Dict, List, Tuple

import pygame
import torch
from PIL import Image

# -------------------------
# Utilities
# -------------------------
def project_root() -> Path:
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


def decode_b64_array(obj: dict) -> torch.Tensor:
    """
    obj = {"dtype": "float16"/"float32", "shape": [...], "b64": "..."}
    returns torch.FloatTensor with the given shape (CPU).
    """
    dtype = obj["dtype"]
    shape = tuple(obj["shape"])
    raw = base64.b64decode(obj["b64"].encode("ascii"))
    if dtype == "float16":
        import numpy as np

        arr = np.frombuffer(raw, dtype=np.float16).reshape(shape).astype("float32", copy=False)
    elif dtype == "float32":
        import numpy as np

        arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
    else:
        raise ValueError(f"Unsupported dtype in embeddings file: {dtype}")
    return torch.from_numpy(arr).float()

# -------------------------
# Image upload helpers
# -------------------------
def center_square_crop_pil(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def load_external_image(path: str, img_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = center_square_crop_pil(img)
    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)

    arr = torch.from_numpy(np.array(img)).float() / 255.0
    return arr.permute(2, 0, 1).contiguous()  # (3,H,W)


# -------------------------
# Editing objective helpers (Method OPT)
# -------------------------
def tv_loss(img_bchw: torch.Tensor) -> torch.Tensor:
    """
    Total variation loss to suppress high-frequency artifacts.
    img_bchw: (B,3,H,W)
    """
    return (img_bchw[:, :, :, 1:] - img_bchw[:, :, :, :-1]).abs().mean() + (img_bchw[:, :, 1:, :] - img_bchw[:, :, :-1, :]).abs().mean()


def soft_clamp_to_stats(z: torch.Tensor, z0: torch.Tensor, k: float = 3.0) -> torch.Tensor:
    """
    Keep z from drifting too far off-manifold by softly clamping per-channel deviations.
    z, z0: (1,C,h,w)
    k: clamp to +/- k * std (per-channel)
    """
    with torch.no_grad():
        sigma = z0.std(dim=(2, 3), keepdim=True) + 1e-6
        lo = z0 - k * sigma
        hi = z0 + k * sigma
    return torch.max(torch.min(z, hi), lo)


def edit_latent_to_type(
    *,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    type_head: torch.nn.Module,
    x_in: torch.Tensor,        # (3,H,W) CPU or GPU
    type_idx: int,
    strength: float,           # signed; >0 push toward type, <0 push away
    steps: int = 30,
    lr: float = 0.05,
    lam_z: float = 1.0,        # keep z near z0
    lam_img_l1: float = 0.5,   # keep decoded image near y0 (prevents adversarial texture)
    lam_tv: float = 0.05,      # suppress high-frequency artifacts
    clamp_k: float = 3.0,      # clamp z within +/-k std of z0 per channel
) -> torch.Tensor:
    """
    Returns edited image y: (3,H,W) on CPU, float in [0,1].

    Per-image latent optimization:
      maximize/minimize classifier score while keeping z close to z0 and output close to original.
    """
    device = next(encoder.parameters()).device
    x = x_in.unsqueeze(0).to(device)  # (1,3,H,W)

    encoder.eval()
    decoder.eval()
    type_head.eval()

    with torch.no_grad():
        z0 = encoder(x)                # (1,C,h,w)
        y0 = decoder(z0).detach()      # (1,3,H,W)

    z = z0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)

    sign = 1.0 if strength >= 0 else -1.0
    target_scale = float(abs(strength))

    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)

        logits = type_head(z)              # (1,num_types)
        score = logits[0, type_idx]        # scalar
        y = decoder(z)

        loss_cls = (-sign * target_scale * score)
        loss_z = lam_z * (z - z0).pow(2).mean()
        loss_img = lam_img_l1 * (y - y0).abs().mean()
        loss_tv_v = lam_tv * tv_loss(y)

        loss = loss_cls + loss_z + loss_img + loss_tv_v
        loss.backward()
        opt.step()

        with torch.no_grad():
            z.copy_(soft_clamp_to_stats(z, z0, k=clamp_k))

    with torch.no_grad():
        y = decoder(z).clamp(0, 1).squeeze(0).detach().cpu()

    return y


# -------------------------
# Fast delta edit (Method DELTA)
# -------------------------
def apply_delta_edit(
    *,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    x_in: torch.Tensor,         # (3,H,W)
    delta_c: torch.Tensor,      # (C,) CPU or GPU
    strength: float,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Fast edit: z' = z + strength * scale * delta[:,None,None]
    Returns y: (3,H,W) CPU float in [0,1]
    """
    device = next(encoder.parameters()).device
    x = x_in.unsqueeze(0).to(device)

    with torch.no_grad():
        z = encoder(x)  # (1,C,h,w)
        d = delta_c.to(device)
        z2 = z + (float(strength) * float(scale)) * d[None, :, None, None]
        y = decoder(z2).clamp(0, 1).squeeze(0).detach().cpu()

    return y


# -------------------------
# Minimal UI widgets (dropdown + slider)
# -------------------------
class Dropdown:
    def __init__(self, rect: pygame.Rect, items: List[str], font: pygame.font.Font, small: pygame.font.Font):
        self.rect = rect
        self.items = items
        self.font = font
        self.small = small

        self.open = False
        self.selected_index = 0

        self.scroll = 0
        self.max_visible = 10  # number of items in open list

    @property
    def value(self) -> str:
        if not self.items:
            return ""
        return self.items[self.selected_index]

    def _scrollbar_rect(self) -> pygame.Rect:
        list_rect = self._list_rect()
        return pygame.Rect(list_rect.right - 12, list_rect.y + 4, 10, list_rect.h - 8)

    def _is_over_scrollbar(self, mx: int, my: int) -> bool:
        if len(self.items) <= self.max_visible:
            return False
        return self._scrollbar_rect().collidepoint(mx, my)

    def handle_event(self, ev: pygame.event.Event) -> Tuple[bool, bool]:
        """
        Returns: (changed, clicked_inside)
        changed=True if selection changed.
        """
        changed = False
        clicked_inside = False

        if ev.type == pygame.MOUSEBUTTONDOWN:
            mx, my = ev.pos
            if self.rect.collidepoint(mx, my):
                clicked_inside = True
                self.open = not self.open
                return False, True

            if self.open:
                list_rect = self._list_rect()
                if list_rect.collidepoint(mx, my):
                    clicked_inside = True

                    if ev.button in (4, 5):
                        return False, True

                    if self._is_over_scrollbar(mx, my):
                        return False, True

                    idx = self._index_from_mouse(mx, my)
                    if idx is not None:
                        self.selected_index = idx
                        self.open = False
                        changed = True
                    return changed, True
                else:
                    self.open = False

        if self.open and len(self.items) > self.max_visible:
            list_rect = self._list_rect()
            mx, my = pygame.mouse.get_pos()
            if list_rect.collidepoint(mx, my):
                if ev.type == pygame.MOUSEWHEEL:
                    self.scroll -= ev.y
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    if ev.button == 4:
                        self.scroll -= 1
                    elif ev.button == 5:
                        self.scroll += 1
                self.scroll = max(0, min(self.scroll, len(self.items) - self.max_visible))

        return changed, clicked_inside

    def _list_rect(self) -> pygame.Rect:
        return pygame.Rect(self.rect.x, self.rect.bottom + 4, self.rect.w, self.max_visible * 22 + 8)

    def _index_from_mouse(self, mx: int, my: int) -> int | None:
        list_rect = self._list_rect()
        if self._is_over_scrollbar(mx, my):
            return None

        y0 = list_rect.y + 4
        if my < y0:
            return None
        row = (my - y0) // 22
        if row < 0 or row >= self.max_visible:
            return None
        idx = self.scroll + int(row)
        if idx < 0 or idx >= len(self.items):
            return None
        return idx

    def draw(self, screen: pygame.Surface):
        pygame.draw.rect(screen, (30, 30, 30), self.rect, border_radius=6)
        pygame.draw.rect(screen, (60, 60, 60), self.rect, width=1, border_radius=6)

        label = self.value if self.value else "(no types)"
        surf = self.small.render(label, True, (230, 230, 230))
        screen.blit(surf, (self.rect.x + 10, self.rect.y + (self.rect.h - surf.get_height()) // 2))

        caret = self.small.render("▼" if not self.open else "▲", True, (180, 180, 180))
        screen.blit(caret, (self.rect.right - caret.get_width() - 10, self.rect.y + 6))

        if self.open:
            list_rect = self._list_rect()
            pygame.draw.rect(screen, (22, 22, 22), list_rect, border_radius=6)
            pygame.draw.rect(screen, (60, 60, 60), list_rect, width=1, border_radius=6)

            start = self.scroll
            end = min(len(self.items), start + self.max_visible)
            y = list_rect.y + 4
            for idx in range(start, end):
                item = self.items[idx]
                row_rect = pygame.Rect(list_rect.x + 4, y, list_rect.w - 8, 22)
                if idx == self.selected_index:
                    pygame.draw.rect(screen, (45, 45, 60), row_rect, border_radius=4)
                surf = self.small.render(item, True, (220, 220, 220))
                screen.blit(surf, (row_rect.x + 6, row_rect.y + 3))
                y += 22

            if len(self.items) > self.max_visible:
                track = self._scrollbar_rect()
                pygame.draw.rect(screen, (40, 40, 40), track, border_radius=3)

                frac = self.max_visible / len(self.items)
                knob_h = max(12, int(track.h * frac))
                max_scroll = len(self.items) - self.max_visible
                t = 0.0 if max_scroll <= 0 else (self.scroll / max_scroll)
                knob_y = track.y + int((track.h - knob_h) * t)
                knob = pygame.Rect(track.x, knob_y, track.w, knob_h)
                pygame.draw.rect(screen, (90, 90, 90), knob, border_radius=3)

# -------------------------
# Simple Button
# -------------------------
class Button:
    def __init__(self, rect: pygame.Rect, text: str, font: pygame.font.Font):
        self.rect = rect
        self.text = text
        self.font = font

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(ev.pos):
                return True
        return False

    def draw(self, screen):
        pygame.draw.rect(screen, (40, 40, 40), self.rect, border_radius=6)
        pygame.draw.rect(screen, (80, 80, 80), self.rect, 1, border_radius=6)

        surf = self.font.render(self.text, True, (220, 220, 220))
        screen.blit(
            surf,
            (
                self.rect.x + (self.rect.w - surf.get_width()) // 2,
                self.rect.y + (self.rect.h - surf.get_height()) // 2,
            ),
        )

class StepSlider:
    def __init__(self, rect: pygame.Rect, steps: List[float], font: pygame.font.Font, small: pygame.font.Font):
        self.rect = rect
        self.steps = steps
        self.font = font
        self.small = small

        self.step_index = steps.index(0.0) if 0.0 in steps else 0
        self.dragging = False

    @property
    def value(self) -> float:
        return float(self.steps[self.step_index])

    def set_value(self, v: float):
        best = min(range(len(self.steps)), key=lambda i: abs(self.steps[i] - v))
        self.step_index = best

    def handle_event(self, ev: pygame.event.Event) -> bool:
        changed = False

        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_LEFT:
                old = self.step_index
                self.step_index = max(0, self.step_index - 1)
                changed = (self.step_index != old)
            elif ev.key == pygame.K_RIGHT:
                old = self.step_index
                self.step_index = min(len(self.steps) - 1, self.step_index + 1)
                changed = (self.step_index != old)

        if ev.type == pygame.MOUSEBUTTONDOWN:
            mx, my = ev.pos
            if self.rect.collidepoint(mx, my):
                self.dragging = True
                changed = self._update_from_mouse(mx)

        if ev.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        if ev.type == pygame.MOUSEMOTION and self.dragging:
            mx, _ = ev.pos
            changed = self._update_from_mouse(mx)

        return changed

    def _update_from_mouse(self, mx: int) -> bool:
        t = (mx - self.rect.x) / max(1, self.rect.w)
        t = max(0.0, min(1.0, t))
        idx = int(round(t * (len(self.steps) - 1)))
        old = self.step_index
        self.step_index = idx
        return self.step_index != old

    def draw(self, screen: pygame.Surface):
        pygame.draw.rect(screen, (30, 30, 30), self.rect, border_radius=6)
        pygame.draw.rect(screen, (60, 60, 60), self.rect, width=1, border_radius=6)

        for i in range(len(self.steps)):
            x = self.rect.x + int(i * (self.rect.w / (len(self.steps) - 1)))
            pygame.draw.line(screen, (55, 55, 55), (x, self.rect.y + self.rect.h - 10), (x, self.rect.y + self.rect.h - 4), 1)

        kx = self.rect.x + int(self.step_index * (self.rect.w / (len(self.steps) - 1)))
        ky = self.rect.y + self.rect.h // 2
        pygame.draw.circle(screen, (200, 200, 220), (kx, ky), 8)
        pygame.draw.circle(screen, (40, 40, 40), (kx, ky), 8, 1)

        label = f"strength: {self.value:+.1f}"
        surf = self.small.render(label, True, (230, 230, 230))
        screen.blit(surf, (self.rect.x, self.rect.y - surf.get_height() - 6))


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="MTG Latent Manipulator (pygame): optimizer edit or saved-delta edit.")
    ap.add_argument("--index", type=str, default="data/index.jsonl", help="Index file (relative to project root).")
    ap.add_argument("--model", type=str, default="models/model.pt", help="AE checkpoint .pt (relative to project root).")
    ap.add_argument("--embeddings", type=str, default="data/creature_embeddings.json", help="Embeddings JSON (vector deltas).")

    ap.add_argument("--window_w", type=int, default=1400)
    ap.add_argument("--window_h", type=int, default=760)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--initial_type", type=str, default="Shapeshifter", help="Initial creature type if present.")

    ap.add_argument(
        "--mode",
        type=str,
        default="delta",
        choices=["opt", "delta"],
        help="opt = classifier-guided latent optimization (slow). delta = apply saved embedding delta vector (fast).",
    )
    ap.add_argument(
        "--delta_key",
        type=str,
        default="delta",
        choices=["delta", "delta_mean_w", "delta_grad", "delta_hybrid"],
        help="Which delta field to use from embeddings JSON when --mode delta.",
    )
    ap.add_argument("--delta_scale", type=float, default=1.0, help="Extra multiplier for delta mode edits.")

    # Method OPT params
    ap.add_argument("--steps", type=int, default=30, help="Optimization steps per render (opt mode).")
    ap.add_argument("--lr", type=float, default=0.05, help="Adam learning rate for z optimization (opt mode).")
    ap.add_argument("--lam_z", type=float, default=1.0, help="L2(z-z0) weight (opt mode).")
    ap.add_argument("--lam_img", type=float, default=0.5, help="L1(y-y0) weight (opt mode).")
    ap.add_argument("--lam_tv", type=float, default=0.05, help="TV loss weight (opt mode).")
    ap.add_argument("--clamp_k", type=float, default=3.0, help="Clamp z within +/-k std of z0 per channel (opt mode).")

    args = ap.parse_args()

    ensure_scripts_on_path()

    from dataloader import MTGCreatureArtDataset
    from create_autoencoder_model import AEConfig, Autoencoder

    root = project_root()
    index_path = (root / args.index).resolve()
    model_path = (root / args.model).resolve()
    emb_path = (root / args.embeddings).resolve()

    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")

    payload = torch.load(str(model_path), map_location="cpu", weights_only=True)
    if not (isinstance(payload, dict) and "config" in payload and "state_dict" in payload):
        raise RuntimeError("Checkpoint format unexpected. Expected dict with keys: config, state_dict.")

    cfg = AEConfig(**payload["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder(cfg).to(device)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        print(f"[INFO] load_state_dict strict=False | missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    encoder = model.encoder
    decoder = model.decoder
    type_head = model.type_head

    # Load embeddings JSON
    emb = json.loads(emb_path.read_text(encoding="utf-8"))
    per_type = emb["per_type"]
    type_names = sorted(per_type.keys())
    if not type_names:
        raise RuntimeError("Embeddings file has no per_type entries.")

    latent_shape = tuple(emb.get("meta", {}).get("latent_shape", []))  # expected (C,)
    if len(latent_shape) != 1:
        print(f"[WARN] embeddings latent_shape={latent_shape} is not (C,). Delta mode expects vector deltas.", flush=True)

    # IMPORTANT: type index mapping must match training vocab order (sorted unique types from index.jsonl)
    def build_vocab_from_index(index_p: Path) -> List[str]:
        types = set()
        with index_p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                cts = e.get("creature_types") or []
                for t in cts:
                    if isinstance(t, str) and t:
                        types.add(t)
        return sorted(types)

    

    def upload_image():
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        root.destroy()

        if not path:
            return

        try:
            x = load_external_image(path, cfg.img_size)
        except Exception as e:
            print(f"[ERROR] failed to load image: {e}")
            return

        edit_cache.clear()

        current["x_in"] = x
        current["title"] = Path(path).name
        current["types"] = "(external image)"
        current["path"] = path
        current["external"] = True

        render_current()

    vocab = build_vocab_from_index(index_path)
    type_to_idx = {t: i for i, t in enumerate(vocab)}

    initial_idx = type_names.index(args.initial_type) if args.initial_type in type_names else 0

    # cache deltas (CPU float32)
    delta_cache: Dict[Tuple[str, str], torch.Tensor] = {}

    def get_delta_tensor(tname: str, dkey: str) -> torch.Tensor:
        ck = (tname, dkey)
        if ck in delta_cache:
            return delta_cache[ck]
        obj = per_type[tname].get(dkey, None)
        if obj is None:
            raise KeyError(f"per_type[{tname}] has no '{dkey}' in embeddings.")
        d = decode_b64_array(obj)  # CPU float32
        if d.ndim != 1:
            raise RuntimeError(f"Delta for {tname}/{dkey} must be 1D (C,). Got {tuple(d.shape)}")
        if len(latent_shape) == 1 and int(d.shape[0]) != int(latent_shape[0]):
            raise RuntimeError(f"Delta C mismatch for {tname}/{dkey}: got {d.shape[0]} expected {latent_shape[0]}")
        delta_cache[ck] = d
        return d

    ds = MTGCreatureArtDataset(
        index_path=index_path,
        img_size=cfg.img_size,
        flip_p=0.0,
        denoise=False,
        seed=args.seed,
        limit=0,
    )
    rng = random.Random(args.seed)

    indices = list(range(len(ds)))
    rng.shuffle(indices)
    ptr = 0

    pygame.init()
    try:
        W, H = args.window_w, args.window_h
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("MTG Latent Manipulator (opt|delta)  (SPACE next, ESC quit)")
        clock = pygame.time.Clock()

        font = pygame.font.SysFont(None, 26)
        small = pygame.font.SysFont(None, 20)

        pad = 14
        top_bar = 130
        bottom_bar = 40
        content_h = H - top_bar - bottom_bar - pad
        panel_w = (W - pad * 3) // 2
        panel_h = content_h

        left_rect = pygame.Rect(pad, top_bar, panel_w, panel_h)
        right_rect = pygame.Rect(pad * 2 + panel_w, top_bar, panel_w, panel_h)

        upload_btn = Button(
            pygame.Rect(left_rect.x + 10, left_rect.y + 10, 100, 30),
            "Upload",
            small,
        )

        dd_rect = pygame.Rect(pad, 70, 360, 30)
        slider_rect = pygame.Rect(pad + 420, 76, 520, 20)

        dropdown = Dropdown(dd_rect, type_names, font=small, small=small)
        dropdown.selected_index = initial_idx

        # NOTE: deltas are normalized in embeddings file; this range might be strong.
        #steps = [round(-5.0 + i * 0.5, 1) for i in range(int((5.0 - (-5.0)) / 0.5) + 1)]
        steps = [round(-10.0 + i * 0.5, 1) for i in range(int((10.0 - (-10.0)) / 0.5) + 1)]
        slider = StepSlider(slider_rect, steps=steps, font=small, small=small)
        slider.set_value(0.0)

        current = {
            "surf_in": None,
            "surf_out": None,
            "title": "",
            "types": "",
            "path": "",
            "x_in": None,  # torch (3,H,W)
        }

        # Cache edited outputs to keep UI responsive
        edit_cache: Dict[Tuple[str, str, float, str, str, float], pygame.Surface] = {}

        def render_current():
            """
            Two modes:
              opt   : classifier-guided per-image latent optimization (slow)
              delta : fast edit using saved per-type delta vectors (C,)
            """
            x_in = current["x_in"]
            if x_in is None:
                return

            tname = dropdown.value
            strength = slider.value

            cache_key = (current["title"], tname, float(strength), args.mode, args.delta_key, float(args.delta_scale))
            if cache_key in edit_cache:
                current["surf_out"] = edit_cache[cache_key]
            else:
                if args.mode == "opt":
                    if tname not in type_to_idx:
                        y = x_in.detach().clone()
                    else:
                        y = edit_latent_to_type(
                            encoder=encoder,
                            decoder=decoder,
                            type_head=type_head,
                            x_in=x_in,
                            type_idx=type_to_idx[tname],
                            strength=strength,
                            steps=args.steps,
                            lr=args.lr,
                            lam_z=args.lam_z,
                            lam_img_l1=args.lam_img,
                            lam_tv=args.lam_tv,
                            clamp_k=args.clamp_k,
                        )
                else:
                    # delta mode
                    if tname not in per_type:
                        y = x_in.detach().clone()
                    else:
                        d = get_delta_tensor(tname, args.delta_key)  # (C,) CPU
                        #y = apply_delta_edit(
                        #    encoder=encoder,
                        #    decoder=decoder,
                        #    x_in=x_in,
                        #    delta_c=d,
                        #    strength=strength,
                        #    scale=args.delta_scale,
                        #)
                        y = x_in.clone()
                        num_steps = 1  # hardcoded for now
                        for _ in range(num_steps):
                            y = apply_delta_edit(
                                encoder=encoder,
                                decoder=decoder,
                                x_in=y,
                                delta_c=d,
                                strength=strength,
                                scale=args.delta_scale,
                            )

                pil_out = tensor_to_pil_rgb(y)
                pil_out_fit = fit_image(pil_out, right_rect.w, right_rect.h)
                surf_out = pil_to_surface(pil_out_fit)
                edit_cache[cache_key] = surf_out
                current["surf_out"] = surf_out

            pil_in = tensor_to_pil_rgb(x_in)
            pil_in_fit = fit_image(pil_in, left_rect.w, left_rect.h)
            current["surf_in"] = pil_to_surface(pil_in_fit)

        def pick_new():
            current["external"] = False
            nonlocal ptr, indices

            if ptr >= len(indices):
                rng.shuffle(indices)   # reshuffle after one full pass
                ptr = 0

            idx = indices[ptr]
            ptr += 1

            sample = ds[idx]

            edit_cache.clear()

            x_in = sample["x_in"]
            creature_types = sample.get("creature_types", [])
            img_path = sample.get("image_path", "")

            current["x_in"] = x_in
            current["title"] = Path(img_path).name if img_path else f"idx={idx}"
            current["types"] = ", ".join(creature_types) if creature_types else "(no types)"
            current["path"] = img_path

            render_current()

        pick_new()

        running = True
        while running:
            rerender = False

            for ev in pygame.event.get():
                if upload_btn.handle_event(ev):
                    upload_image()

                if ev.type == pygame.QUIT:
                    running = False

                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_SPACE:
                        pick_new()
                    if slider.handle_event(ev):
                        rerender = True

                else:
                    changed, _clicked_inside = dropdown.handle_event(ev)
                    if changed:
                        slider.set_value(0.0)
                        rerender = True
                    if slider.handle_event(ev):
                        rerender = True

            if rerender:
                render_current()

            screen.fill((10, 10, 10))

            if args.mode == "opt":
                title = "Mode OPT: optimize z to change selected type logit"
            else:
                title = f"Mode DELTA: z' = z + strength*delta (key={args.delta_key}, scale={args.delta_scale})"

            draw_text(screen, font, title, pad, 12)
            draw_text(screen, small, f"File: {current['title']}", pad, 28, color=(170, 170, 170))

            slider.draw(screen)

            if args.mode == "opt":
                draw_text(
                    screen,
                    small,
                    f"steps={args.steps} lr={args.lr}  lam_z={args.lam_z} lam_img={args.lam_img} lam_tv={args.lam_tv} clamp_k={args.clamp_k}",
                    pad + 900,
                    52,
                    color=(160, 160, 160),
                )
            else:
                draw_text(
                    screen,
                    small,
                    f"delta_key={args.delta_key}  delta_scale={args.delta_scale}",
                    pad + 900,
                    52,
                    color=(160, 160, 160),
                )

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

            upload_btn.draw(screen)

            draw_text(screen, small, "Creature type:", dd_rect.x, dd_rect.y - 18, color=(190, 190, 190))
            dropdown.draw(screen)

            info_y = H - 40
            pygame.draw.rect(screen, (15, 15, 15), pygame.Rect(0, info_y, W, 40))
            draw_text(screen, small, f"Card types: {current['types']}", pad, info_y + 10, color=(200, 200, 200))

            hint = "SPACE = next random    |    ESC = quit"
            hint_surf = small.render(hint, True, (150, 150, 150))
            screen.blit(hint_surf, (W - hint_surf.get_width() - pad, info_y + 10))

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