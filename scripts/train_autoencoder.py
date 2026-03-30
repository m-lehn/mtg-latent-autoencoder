from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- make imports work when running from project root ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Local modules
from create_autoencoder_model import AEConfig, Autoencoder  # type: ignore
from dataloader import DataConfig, make_dataloader  # type: ignore

from loss import (  # type: ignore
    LossWeights,
    CombinedLoss,
    CharbonnierLoss,
    L1Loss,
    MSELoss,
    EdgeLoss,
    PatchCriticDegradeLoss,
    TotalVariationLoss,
    PerceptualLossVGG,
    CreatureTypeBCELoss,
)


# -------------------------
# Helpers
# -------------------------
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_torch_load(path: Path, map_location="cpu"):
    """
    Avoid FutureWarning where possible, but stay compatible across torch versions.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def write_jsonl(path: Path, entries: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    entries: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def split_index_jsonl(
    index_path: Path, out_dir: Path, val_ratio: float, seed: int
) -> Tuple[Path, Path, int, int]:
    entries = read_jsonl(index_path)
    if not entries:
        raise RuntimeError(f"No entries in index: {index_path}")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(entries), generator=g).tolist()

    n_val = int(round(len(entries) * val_ratio))
    n_val = max(1, n_val)
    n_train = max(1, len(entries) - n_val)

    val_ids = perm[:n_val]
    train_ids = perm[n_val:]

    train_entries = [entries[i] for i in train_ids]
    val_entries = [entries[i] for i in val_ids]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    write_jsonl(train_path, train_entries)
    write_jsonl(val_path, val_entries)

    return train_path, val_path, len(train_entries), len(val_entries)


def save_preview_grid(
    run_dir: Path,
    epoch: int,
    x_in: torch.Tensor,
    x_gt: torch.Tensor,
    y: torch.Tensor,
    max_items: int = 6,
) -> None:
    """
    Saves preview image: per row [input | output | target]
    """
    try:
        from torchvision.utils import save_image
    except Exception as e:
        print(f"[WARN] torchvision not available for previews: {e}")
        return

    x_in = x_in.detach().cpu().clamp(0, 1)
    x_gt = x_gt.detach().cpu().clamp(0, 1)
    y = y.detach().cpu().clamp(0, 1)

    b = min(x_in.size(0), max_items)
    triplets = torch.cat([x_in[:b], y[:b], x_gt[:b]], dim=3)

    out_path = run_dir / f"preview_epoch{epoch:03d}.png"
    save_image(triplets, out_path, nrow=1)


def tensor_stats(x: torch.Tensor) -> str:
    return (
        f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
        f"min={x.min().item():.4f} max={x.max().item():.4f} "
        f"mean={x.mean().item():.4f} std={x.std().item():.4f} "
        f"finite={torch.isfinite(x).all().item()}"
    )


def build_type_vocab(index_path: Path) -> list[str]:
    types = set()
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            for t in (e.get("creature_types") or []):
                if isinstance(t, str) and t:
                    types.add(t)
    if not types:
        raise RuntimeError(f"No creature_types found in {index_path}")
    return sorted(types)


def make_multihot(
    batch_types: list[list[str]],
    type_to_idx: dict[str, int],
    num_types: int,
    device: torch.device,
) -> torch.Tensor:
    y = torch.zeros((len(batch_types), num_types), dtype=torch.float32, device=device)
    for i, types in enumerate(batch_types):
        for t in (types or []):
            j = type_to_idx.get(t)
            if j is not None:
                y[i, j] = 1.0
    return y


def compute_pos_weight_from_index(
    index_path: Path,
    type_to_idx: dict[str, int],
    num_types: int,
) -> torch.Tensor:
    pos_counts = torch.zeros((num_types,), dtype=torch.float32)
    n = 0
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            n += 1
            for t in (e.get("creature_types") or []):
                j = type_to_idx.get(t)
                if j is not None:
                    pos_counts[j] += 1.0

    neg_counts = float(n) - pos_counts
    pos_weight = neg_counts / (pos_counts + 1.0)  # +1 avoids inf for ultra-rare
    return pos_weight


# -------------------------
# Loss builder (image-only)
# -------------------------
def build_image_loss(args) -> CombinedLoss:
    # pixel loss
    if args.pixel_loss == "l1":
        pixel = L1Loss()
    elif args.pixel_loss == "mse":
        pixel = MSELoss()
    else:
        pixel = CharbonnierLoss(eps=args.charb_eps)

    perceptual = None
    if args.w_perc > 0:
        perceptual = PerceptualLossVGG(
            layers=("relu2_2", "relu3_3"),
            use_l1=True,
        )

    edge = None
    if args.w_edge > 0:
        edge = EdgeLoss(use_luma=True, loss="l1")

    patch_critic = None
    if args.w_critic > 0:
        patch_critic = PatchCriticDegradeLoss(ckpt_path="models/patch_critic.pt")

    tv = None
    if args.w_tv > 0:
        tv = TotalVariationLoss()

    weights = LossWeights(
        pixel=1.0,
        perceptual=args.w_perc,
        edge=args.w_edge,
        tv=args.w_tv,
        patch_critic=args.w_critic,
        cls=args.w_cls,  # NOTE: renamed in your loss.py
    )

    # IMPORTANT: CombinedLoss here is image-only.
    return CombinedLoss(
        pixel_loss=pixel,
        weights=weights,
        perceptual_loss=perceptual,
        edge_loss=edge,
        tv_loss=tv,
        patch_critic_loss=patch_critic,
    )


# -------------------------
# Train / Val
# -------------------------
@torch.no_grad()
def run_validation(
    model: nn.Module,
    dl: DataLoader,
    img_criterion: CombinedLoss,
    cls_loss_fn: nn.Module | None,
    type_to_idx: dict[str, int],
    num_types: int,
    device: torch.device,
    use_amp: bool,
    w_cls: float,
) -> Dict[str, float]:
    model.eval()

    totals: Dict[str, float] = {}
    n = 0

    for batch in dl:
        x_in = batch["x_in"].to(device, non_blocking=True)
        x_gt = batch["x_gt"].to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                x_hat, type_logits = model(x_in)  # ALWAYS returns both heads
                loss_img, comps = img_criterion(x_hat, x_gt)
                loss_total = loss_img

                if w_cls > 0 and cls_loss_fn is not None:
                    y_cls = make_multihot(batch["creature_types"], type_to_idx, num_types, device)
                    l_cls = cls_loss_fn(type_logits, y_cls)
                    loss_total = loss_total + (w_cls * l_cls)
                    comps["cls"] = float(l_cls.detach().item())

                comps["total"] = float(loss_total.detach().item())
        else:
            x_hat, type_logits = model(x_in)
            loss_img, comps = img_criterion(x_hat, x_gt)
            loss_total = loss_img

            if w_cls > 0 and cls_loss_fn is not None:
                y_cls = make_multihot(batch["creature_types"], type_to_idx, num_types, device)
                l_cls = cls_loss_fn(type_logits, y_cls)
                loss_total = loss_total + (w_cls * l_cls)
                comps["cls"] = float(l_cls.detach().item())

            comps["total"] = float(loss_total.detach().item())

        for k, v in comps.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n += 1

    if n == 0:
        return {"total": float("nan")}
    return {k: v / n for k, v in totals.items()}


def train_one_epoch(
    model: nn.Module,
    dl: DataLoader,
    img_criterion: CombinedLoss,
    cls_loss_fn: nn.Module | None,
    type_to_idx: dict[str, int],
    num_types: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
    log_every: int,
    epoch: int,
    w_cls: float,
) -> Dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    totals: Dict[str, float] = {}
    n = 0
    t0 = time.time()

    for step, batch in enumerate(dl):
        x_in = batch["x_in"].to(device, non_blocking=True)
        x_gt = batch["x_gt"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                x_hat, type_logits = model(x_in)  # ALWAYS returns both heads

                loss_img, comps = img_criterion(x_hat, x_gt)
                loss_total = loss_img

                if w_cls > 0 and cls_loss_fn is not None:
                    y_cls = make_multihot(batch["creature_types"], type_to_idx, num_types, device)
                    l_cls = cls_loss_fn(type_logits, y_cls)
                    loss_total = loss_total + (w_cls * l_cls)
                    comps["cls"] = float(l_cls.detach().item())

                comps["total"] = float(loss_total.detach().item())

            scaler.scale(loss_total).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            x_hat, type_logits = model(x_in)

            loss_img, comps = img_criterion(x_hat, x_gt)
            loss_total = loss_img

            if w_cls > 0 and cls_loss_fn is not None:
                y_cls = make_multihot(batch["creature_types"], type_to_idx, num_types, device)
                l_cls = cls_loss_fn(type_logits, y_cls)
                loss_total = loss_total + (w_cls * l_cls)
                comps["cls"] = float(l_cls.detach().item())

            comps["total"] = float(loss_total.detach().item())

            loss_total.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        for k, v in comps.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n += 1

        if (step % log_every) == 0:
            msg = f"epoch {epoch:03d} step {step:04d} loss {comps.get('total', float('nan')):.6f}"
            if "pixel" in comps:
                msg += f" | pix {comps['pixel']:.6f}"
            if "perceptual" in comps:
                msg += f" perc {comps['perceptual']:.6f}"
            if "edge" in comps:
                msg += f" edge {comps['edge']:.6f}"
            if "tv" in comps:
                msg += f" tv {comps['tv']:.6f}"
            if "patch_critic" in comps:
                msg += f" critic {comps['patch_critic']:.6f}"
            if "cls" in comps:
                msg += f" cls {comps['cls']:.6f}"
            print(msg)

    dt = time.time() - t0
    out = {k: v / max(1, n) for k, v in totals.items()}
    out["sec"] = dt
    return out


# -------------------------
# Checkpointing
# -------------------------
def save_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: AEConfig,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    payload = {
        "config": asdict(cfg),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(payload, path)


def load_model_from_pt(pt_path: Path) -> Tuple[Autoencoder, AEConfig, Dict]:
    payload = safe_torch_load(pt_path, map_location="cpu")
    cfg_dict = payload.get("config", {})
    cfg = AEConfig(**cfg_dict)
    model = Autoencoder(cfg)
    if "state_dict" in payload:
        #model.load_state_dict(payload["state_dict"], strict=True)
        missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
        print(f"[INFO] load_state_dict strict=False | missing={len(missing)} unexpected={len(unexpected)}")
    return model, cfg, payload


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("Train MTG autoencoder + latent classifier")

    ap.add_argument("--model", type=str, default="models/model.pt")
    ap.add_argument("--index", type=str, default="data/index.jsonl")

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--denoise", action="store_true")

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--pixel_loss", type=str, default="charb", choices=["l1", "mse", "charb"])
    ap.add_argument("--charb_eps", type=float, default=1e-3)

    ap.add_argument("--w_perc", type=float, default=0.0)
    ap.add_argument("--w_edge", type=float, default=0.0)
    ap.add_argument("--w_tv", type=float, default=0.0)
    ap.add_argument("--w_critic", type=float, default=0.0)

    ap.add_argument("--w_cls", type=float, default=0.0)
    ap.add_argument("--cls_label_smoothing", type=float, default=0.0)
    ap.add_argument("--cls_pos_weight", action="store_true")

    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--preview_every", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=1)

    args = ap.parse_args()

    model_path = (PROJECT_ROOT / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    index_path = (PROJECT_ROOT / args.index).resolve() if not Path(args.index).is_absolute() else Path(args.index)

    ensure_dir(PROJECT_ROOT / "runs")
    ensure_dir(PROJECT_ROOT / "models")

    run_dir = PROJECT_ROOT / "runs" / now_tag()
    ensure_dir(run_dir)

    train_index, val_index, n_train, n_val = split_index_jsonl(
        index_path=index_path,
        out_dir=run_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model, cfg, payload = load_model_from_pt(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    # Build type vocab from FULL index (not only train)
    type_vocab = build_type_vocab(index_path)
    type_to_idx = {t: i for i, t in enumerate(type_vocab)}
    num_types = len(type_vocab)

    cfg_num = getattr(cfg, "num_types", None)
    if cfg_num is not None and cfg_num != num_types:
        print(f"[WARN] cfg.num_types={cfg_num} but index has {num_types} types. "
              f"Classifier head/output must match model config. "
              f"Fix by regenerating model with the same index.")
        # This mismatch is serious: the head dimension must match.
        # We'll hard-stop to avoid silent shape errors.
        raise RuntimeError("num_types mismatch between model config and index vocab.")

    (run_dir / "type_vocab.json").write_text(json.dumps(type_vocab, indent=2), encoding="utf-8")

    # classifier loss (optional)
    cls_loss_fn = None
    if args.w_cls > 0:
        pos_weight = None
        if args.cls_pos_weight:
            pos_weight = compute_pos_weight_from_index(index_path, type_to_idx, num_types).to(device)

        cls_loss_fn = CreatureTypeBCELoss(
            label_smoothing=args.cls_label_smoothing,
            pos_weight=pos_weight,
        ).to(device)

    print("====================================")
    print("Device:", device, "| AMP:", use_amp)
    print("Project root:", PROJECT_ROOT)
    print("Run dir:", run_dir)
    print("Model path:", model_path)
    print("Index path:", index_path)
    print(f"Train/Val: {n_train} / {n_val}")
    print("Model cfg:", json.dumps(asdict(cfg), indent=2))
    print("Train cfg:", json.dumps({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "denoise": args.denoise,
        "pixel_loss": args.pixel_loss,
        "charb_eps": args.charb_eps,
        "w_perc": args.w_perc,
        "w_edge": args.w_edge,
        "w_tv": args.w_tv,
        "w_critic": args.w_critic,
        "grad_clip": args.grad_clip,
        "w_cls": args.w_cls,
        "cls_label_smoothing": args.cls_label_smoothing,
        "cls_pos_weight": args.cls_pos_weight,
    }, indent=2))
    print("====================================")

    model = model.to(device)

    train_dl = make_dataloader(DataConfig(
        index_path=str(train_index.relative_to(PROJECT_ROOT)),
        img_size=cfg.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        denoise=args.denoise,
        flip_p=0.5,
        limit=0,
    ), seed=args.seed)

    val_dl = make_dataloader(DataConfig(
        index_path=str(val_index.relative_to(PROJECT_ROOT)),
        img_size=cfg.img_size,
        batch_size=args.batch_size,
        num_workers=max(0, args.num_workers // 2),
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        denoise=False,
        flip_p=0.0,
        limit=0,
    ), seed=args.seed + 999)

    img_criterion = build_image_loss(args).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if "optimizer" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer"])
            print("[INFO] Loaded optimizer state from checkpoint.")
        except Exception as e:
            print("[WARN] Could not load optimizer state:", e)

    start_epoch = int(payload.get("epoch", -1)) + 1 if "epoch" in payload else 0

    csv_path = run_dir / "log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "train_total", "train_pixel", "train_perc", "train_edge", "train_tv", "train_critic", "train_cls",
            "val_total", "val_pixel", "val_perc", "val_edge", "val_tv", "val_critic", "val_cls",
            "sec"
        ])

    batch0 = next(iter(train_dl))
    print("[SANITY] x_in:", tensor_stats(batch0["x_in"]))
    print("[SANITY] x_gt:", tensor_stats(batch0["x_gt"]))

    preview_iter = iter(train_dl)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        t_train = train_one_epoch(
            model=model,
            dl=train_dl,
            img_criterion=img_criterion,
            cls_loss_fn=cls_loss_fn,
            type_to_idx=type_to_idx,
            num_types=num_types,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            epoch=epoch,
            w_cls=args.w_cls,
        )

        v = run_validation(
            model=model,
            dl=val_dl,
            img_criterion=img_criterion,
            cls_loss_fn=cls_loss_fn,
            type_to_idx=type_to_idx,
            num_types=num_types,
            device=device,
            use_amp=use_amp,
            w_cls=args.w_cls,
        )

        # Preview Block
        if args.preview_every > 0 and (epoch % args.preview_every) == 0:
            model.eval()
            with torch.no_grad():
                try:
                    b = next(preview_iter)
                except StopIteration:
                    preview_iter = iter(train_dl)
                    b = next(preview_iter)
                x_in = b["x_in"].to(device, non_blocking=True)
                x_gt = b["x_gt"].to(device, non_blocking=True)
                if use_amp and device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        y, _logits = model(x_in)
                else:
                    y, _logits = model(x_in)
            save_preview_grid(run_dir, epoch, x_in, x_gt, y)

        if args.save_every > 0 and (epoch % args.save_every) == 0:
            save_checkpoint(model_path, model, cfg, optimizer, epoch)

        def g(d, k): return float(d.get(k, float("nan")))

        print(
            f"Epoch {epoch:03d} done | "
            f"train {g(t_train,'total'):.6f} (pix {g(t_train,'pixel'):.6f} edge {g(t_train,'edge'):.6f} tv {g(t_train,'tv'):.6f} critic {g(t_train,'patch_critic'):.6f} cls {g(t_train,'cls'):.6f}) | "
            f"val {g(v,'total'):.6f} (pix {g(v,'pixel'):.6f} edge {g(v,'edge'):.6f} tv {g(v,'tv'):.6f} critic {g(v,'patch_critic'):.6f} cls {g(v,'cls'):.6f}) | "
            f"{t_train.get('sec',0.0):.1f}s"
        )

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                epoch,
                g(t_train, "total"), g(t_train, "pixel"), g(t_train, "perceptual"), g(t_train, "edge"),
                g(t_train, "tv"), g(t_train, "patch_critic"), g(t_train, "cls"),
                g(v, "total"), g(v, "pixel"), g(v, "perceptual"), g(v, "edge"),
                g(v, "tv"), g(v, "patch_critic"), g(v, "cls"),
                float(t_train.get("sec", 0.0)),
            ])

    save_checkpoint(model_path, model, cfg, optimizer, epoch)
    print("Training finished. Final checkpoint saved to:", model_path)
    print("Run logs:", run_dir)


if __name__ == "__main__":
    main()