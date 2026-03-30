# scripts/save_embeddings.py
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


# -------------------------
# Paths / imports
# -------------------------
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_scripts_on_path():
    scripts_dir = project_root() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


# -------------------------
# Debug helpers
# -------------------------
def now() -> float:
    return time.perf_counter()


def fmt_s(dt_s: float) -> str:
    if dt_s < 1:
        return f"{dt_s*1000:.0f}ms"
    if dt_s < 60:
        return f"{dt_s:.1f}s"
    return f"{dt_s/60:.1f}m"


def dbg(msg: str, enabled: bool) -> None:
    if enabled:
        print(msg, flush=True)


def cuda_mem_str() -> str:
    if not torch.cuda.is_available():
        return "cuda: not available"
    try:
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserv = torch.cuda.memory_reserved() / 1024**2
        max_alloc = torch.cuda.max_memory_allocated() / 1024**2
        max_reserv = torch.cuda.max_memory_reserved() / 1024**2
        return f"cuda MiB alloc={alloc:.0f} reserv={reserv:.0f} max_alloc={max_alloc:.0f} max_reserv={max_reserv:.0f}"
    except Exception as e:
        return f"cuda mem: error({type(e).__name__}: {e})"


# -------------------------
# Base64 helpers
# -------------------------
def encode_array_to_b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


# -------------------------
# Build deterministic creature-type vocab from index.jsonl
# (must match training ordering!)
# -------------------------
def build_type_vocab(index_path: Path) -> List[str]:
    types = set()
    with index_path.open("r", encoding="utf-8") as f:
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


# -------------------------
# Collate
# -------------------------
def mtg_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["creature_types"] = [b.get("creature_types", []) for b in batch]
    out["image_path"] = [b.get("image_path", "") for b in batch]

    # preferred: dataset returns deterministic views
    if "x_views" in batch[0]:
        out["x_views"] = torch.stack([b["x_views"] for b in batch], dim=0)  # (B,4,3,H,W)
        return out

    # fallback: dataset returns separate keys
    keys4 = ("x_rgb", "x_rgb_flip", "x_gray", "x_gray_flip")
    if all(k in batch[0] for k in keys4):
        views = []
        for b in batch:
            views.append(torch.stack([b["x_rgb"], b["x_rgb_flip"], b["x_gray"], b["x_gray_flip"]], dim=0))
        out["x_views"] = torch.stack(views, dim=0)  # (B,4,3,H,W)
        return out

    # last resort (single view)
    out["x_in"] = torch.stack([b["x_in"] for b in batch], dim=0)
    return out


# -------------------------
# Fallback view ops (only if dataset doesn't support return_views)
# -------------------------
def to_grayscale_3ch(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y.repeat(1, 3, 1, 1)


def hflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[3])


# -------------------------
# Packing helpers
# -------------------------
def pack_tensor(t: torch.Tensor, save_dtype: np.dtype, dtype_name: str) -> Dict[str, Any]:
    arr = t.detach().cpu().numpy().astype(save_dtype, copy=False)
    return {"dtype": dtype_name, "shape": list(arr.shape), "b64": encode_array_to_b64(arr)}


def l2_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (torch.linalg.norm(v) + eps)


def safe_std_from_sums(sum_: torch.Tensor, sumsq: torch.Tensor, n: int, eps: float = 1e-6) -> torch.Tensor:
    # var = E[x^2] - E[x]^2
    mean = sum_ / float(max(n, 1))
    ex2 = sumsq / float(max(n, 1))
    var = (ex2 - mean * mean).clamp_min(0.0)
    return (var + eps).sqrt()


def popcount_types(cts: Any) -> int:
    if not cts:
        return 0
    return sum(1 for t in cts if isinstance(t, str) and t)


@torch.no_grad()
def main():
    print("[boot] save_embeddings.py starting...", flush=True)

    ap = argparse.ArgumentParser(
        description=(
            "Compute per-creature-type edit directions in pooled latent space.\n"
            "Saves CHANNEL-ONLY vectors (C,) to avoid PixelShuffle phase/grid artifacts.\n"
            "Outputs:\n"
            "  delta_mean_w  : whitened mean-diff direction\n"
            "  delta_grad    : pooled gradient direction (optional)\n"
            "  delta_hybrid  : normalized blend of mean+grad\n"
            "  delta         : chosen default (hybrid if grad enabled else mean_w)"
        )
    )
    ap.add_argument("--index", type=str, default="data/index.jsonl")
    ap.add_argument("--model", type=str, default="models/model.pt")
    ap.add_argument("--out", type=str, default="data/creature_embeddings_vec.json")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)  # Windows-safe default
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--max_cards", type=int, default=0)
    ap.add_argument("--min_count", type=int, default=20)

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--save_dtype", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--return_views", action="store_true")

    # classifier-informed direction
    ap.add_argument("--use_grad_dir", action="store_true", help="Also compute pooled gradient directions per type.")
    ap.add_argument("--grad_microbatch", type=int, default=8, help="Microbatch size for grad direction pass (views).")
    ap.add_argument("--grad_abs", action="store_true", help="Use abs(grad) before pooling (more stable, less semantic).")
    ap.add_argument("--grad_eps", type=float, default=1e-8)

    # combining
    ap.add_argument("--hybrid_alpha", type=float, default=0.5, help="Blend: hybrid = normalize((1-a)*mean_w + a*grad).")
    ap.add_argument("--whiten_kind", type=str, default="gray", choices=["rgb", "gray", "all"],
                    help="Which distribution to use for per-channel std whitening.")

    # debug
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--print_every", type=int, default=50)

    # allocator hint (must be set before first CUDA alloc)
    ap.add_argument("--expandable_segments", action="store_true",
                    help="Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (helps fragmentation).")
    args = ap.parse_args()
    debug = bool(args.debug)

    if args.expandable_segments:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    ensure_scripts_on_path()

    from dataloader import MTGCreatureArtDataset
    from create_autoencoder_model import AEConfig, Autoencoder

    root = project_root()
    index_path = (root / args.index).resolve()
    model_path = (root / args.model).resolve()
    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("====================================", flush=True)
    print("Build creature embeddings (CHANNEL VECTORS)", flush=True)
    print(f"Project root : {root}", flush=True)
    print(f"Device       : {device}", flush=True)
    print(f"Index        : {index_path}", flush=True)
    print(f"Checkpoint   : {model_path}", flush=True)
    print(f"batch_size   : {args.batch_size} | workers {args.num_workers} | pin_memory {bool(args.pin_memory)}", flush=True)
    print(f"return_views : {bool(args.return_views)}", flush=True)
    print(f"use_grad_dir : {bool(args.use_grad_dir)} | microbatch {args.grad_microbatch} | grad_abs {bool(args.grad_abs)}", flush=True)
    print(f"hybrid_alpha : {args.hybrid_alpha} | whiten_kind {args.whiten_kind}", flush=True)
    print("====================================", flush=True)

    # -------------------------
    # Load checkpoint + model
    # -------------------------
    payload = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if not (isinstance(payload, dict) and "config" in payload and "state_dict" in payload):
        raise RuntimeError("Checkpoint format unexpected. Expected dict with keys: config, state_dict.")

    cfg = AEConfig(**payload["config"])
    model = Autoencoder(cfg).to(device)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict(strict=False) mismatches | missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if debug and missing:
            print("  missing (first 20):", missing[:20], flush=True)
        if debug and unexpected:
            print("  unexpected (first 20):", unexpected[:20], flush=True)

    model.eval()

    if args.use_grad_dir and getattr(model, "type_head", None) is None:
        raise RuntimeError("--use_grad_dir requested but model has no type_head.")

    # -------------------------
    # Vocab (must match training order)
    # -------------------------
    vocab = build_type_vocab(index_path)
    type_to_idx = {t: i for i, t in enumerate(vocab)}
    num_types = len(vocab)

    # -------------------------
    # Dataset + loader
    # -------------------------
    ds_kwargs = dict(
        index_path=index_path,
        img_size=cfg.img_size,
        flip_p=0.0,      # deterministic views
        denoise=False,
        seed=123,
        limit=0,
    )
    if args.return_views:
        try:
            ds = MTGCreatureArtDataset(**ds_kwargs, return_views=True)
        except TypeError:
            print("[WARN] Dataset does not accept return_views=True. Falling back to single-view mode.", flush=True)
            ds = MTGCreatureArtDataset(**ds_kwargs)
    else:
        ds = MTGCreatureArtDataset(**ds_kwargs)

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory and device.type == "cuda"),
        drop_last=False,
        collate_fn=mtg_collate,
        persistent_workers=False,
    )

    # -------------------------
    # Accumulators (CPU float64)
    # We accumulate pooled latent vectors:
    #   v = mean(z, spatial) -> (C,)
    #
    # overall sums for rgb and gray (each uses 2 views per card)
    # + sumsqs for whitening std
    # per-type sums for rgb/gray and optional grad directions
    # -------------------------
    C = int(cfg.bottleneck_channels)

    overall_sum_rgb = torch.zeros((C,), dtype=torch.float64)
    overall_sumsq_rgb = torch.zeros((C,), dtype=torch.float64)
    overall_count_rgb = 0

    overall_sum_gray = torch.zeros((C,), dtype=torch.float64)
    overall_sumsq_gray = torch.zeros((C,), dtype=torch.float64)
    overall_count_gray = 0

    # all-views whitening option
    overall_sum_all = torch.zeros((C,), dtype=torch.float64)
    overall_sumsq_all = torch.zeros((C,), dtype=torch.float64)
    overall_count_all = 0

    per_type_sum_rgb: Dict[str, torch.Tensor] = {}
    per_type_sum_gray: Dict[str, torch.Tensor] = {}
    per_type_count_rgb: Dict[str, int] = {}
    per_type_count_gray: Dict[str, int] = {}

    # optional pooled gradient direction accumulators (vectors only)
    per_type_sum_grad: Dict[str, torch.Tensor] = {}
    per_type_count_grad: Dict[str, int] = {}

    cards_processed = 0
    t_start = now()

    for step, batch in enumerate(dl):
        types_list = batch["creature_types"]

        # Build views on CPU
        if "x_views" in batch:
            x_views = batch["x_views"]  # (B,4,3,H,W)
        else:
            x_in = batch["x_in"]
            x_views = torch.stack(
                [x_in, hflip(x_in), to_grayscale_3ch(x_in), hflip(to_grayscale_3ch(x_in))],
                dim=1,
            )

        B = int(x_views.shape[0])
        cards_processed += B

        if args.max_cards and cards_processed > args.max_cards:
            keep = max(0, B - (cards_processed - args.max_cards))
            x_views = x_views[:keep]
            types_list = types_list[:keep]
            B = keep
            cards_processed = args.max_cards

        if B == 0:
            break

        # --------------- Pass 1: pooled latent vectors for 4 views (no grad) ---------------
        x4 = x_views.reshape(B * 4, *x_views.shape[2:]).to(device, non_blocking=True)
        z4 = model.encoder(x4)                 # (4B,C,h,w)
        v4 = z4.mean(dim=(2, 3))               # (4B,C) pooled
        v4_cpu = v4.detach().to("cpu", dtype=torch.float64).reshape(B, 4, C)  # (B,4,C)

        # free GPU tensors ASAP
        del x4, z4, v4

        # overall accum (two views each)
        v_rgb = v4_cpu[:, 0:2].reshape(B * 2, C)   # (2B,C)
        v_gray = v4_cpu[:, 2:4].reshape(B * 2, C)  # (2B,C)

        overall_sum_rgb += v_rgb.sum(dim=0)
        overall_sumsq_rgb += (v_rgb * v_rgb).sum(dim=0)
        overall_count_rgb += int(v_rgb.shape[0])

        overall_sum_gray += v_gray.sum(dim=0)
        overall_sumsq_gray += (v_gray * v_gray).sum(dim=0)
        overall_count_gray += int(v_gray.shape[0])

        v_all = v4_cpu.reshape(B * 4, C)
        overall_sum_all += v_all.sum(dim=0)
        overall_sumsq_all += (v_all * v_all).sum(dim=0)
        overall_count_all += int(v_all.shape[0])

        # per-type mean-diff accum (sum two views per card)
        for i in range(B):
            cts = types_list[i] or []
            if not cts:
                continue
            sum_rgb_i = v4_cpu[i, 0:2].sum(dim=0)   # (C,)
            sum_gray_i = v4_cpu[i, 2:4].sum(dim=0)

            for t in cts:
                if not isinstance(t, str) or not t:
                    continue
                if t not in per_type_sum_rgb:
                    per_type_sum_rgb[t] = torch.zeros((C,), dtype=torch.float64)
                    per_type_sum_gray[t] = torch.zeros((C,), dtype=torch.float64)
                    per_type_count_rgb[t] = 0
                    per_type_count_gray[t] = 0
                    if args.use_grad_dir:
                        per_type_sum_grad[t] = torch.zeros((C,), dtype=torch.float64)
                        per_type_count_grad[t] = 0

                per_type_sum_rgb[t] += sum_rgb_i
                per_type_sum_gray[t] += sum_gray_i
                per_type_count_rgb[t] += 2
                per_type_count_gray[t] += 2

        # --------------- Pass 2: pooled gradient directions (optional) ---------------
        if args.use_grad_dir:
            # Build (2B,3,H,W) gray views
            x_gray2_cpu = x_views[:, 2:4].reshape(B * 2, *x_views.shape[2:])

            # Build mask (B,T) -> repeat to (2B,T)
            # We use mask to compute per-view score as mean logit over the card's types.
            mask = torch.zeros((B, num_types), dtype=torch.float32)
            for i in range(B):
                for t in (types_list[i] or []):
                    j = type_to_idx.get(t, None)
                    if j is not None:
                        mask[i, j] = 1.0
            mask2_cpu = mask.repeat_interleave(2, dim=0)  # (2B,T)

            mb = max(1, int(args.grad_microbatch))
            for start in range(0, B * 2, mb):
                end = min(B * 2, start + mb)

                x_mb = x_gray2_cpu[start:end].to(device, non_blocking=True)
                m_mb = mask2_cpu[start:end].to(device, non_blocking=True)

                model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    z = model.encoder(x_mb).float()      # (mb,C,h,w)
                    z.requires_grad_(True)
                    logits = model.type_head(z)          # (mb,T)

                    den = m_mb.sum(dim=1).clamp_min(1.0)
                    s = (logits * m_mb).sum(dim=1) / den  # (mb,)

                    grads = torch.autograd.grad(
                        outputs=s,
                        inputs=z,
                        grad_outputs=torch.ones_like(s),
                        retain_graph=False,
                        create_graph=False,
                        only_inputs=True,
                    )[0]  # (mb,C,h,w)

                    if args.grad_abs:
                        grads = grads.abs()

                    g = grads.mean(dim=(2, 3))  # (mb,C) pooled per view

                    # normalize each view grad to avoid a few samples dominating
                    g = g / (g.norm(dim=1, keepdim=True) + args.grad_eps)

                g_cpu = g.detach().to("cpu", dtype=torch.float64)  # (mb,C)

                # accumulate per CARD (sum its two views) then per type
                for view_idx in range(start, end):
                    card_i = view_idx // 2
                    cts = types_list[card_i] or []
                    if not cts:
                        continue
                    local = view_idx - start
                    g_view = g_cpu[local]  # (C,)

                    for t in cts:
                        if not isinstance(t, str) or not t:
                            continue
                        if t not in per_type_sum_grad:
                            # (should exist from pass1 init, but be safe)
                            per_type_sum_grad[t] = torch.zeros((C,), dtype=torch.float64)
                            per_type_count_grad[t] = 0
                        per_type_sum_grad[t] += g_view
                        per_type_count_grad[t] += 1

                # free GPU tensors promptly
                del x_mb, m_mb, z, logits, den, s, grads, g, g_cpu

            # free CPU staging tensors
            del x_gray2_cpu, mask, mask2_cpu

        if step % max(1, int(args.print_every)) == 0:
            elapsed = now() - t_start
            rate = cards_processed / max(elapsed, 1e-9)
            print(
                f"step {step:05d} | cards {cards_processed} | "
                f"types tracked {len(per_type_sum_rgb)} | "
                f"{rate:.1f} cards/s | {fmt_s(elapsed)} elapsed",
                flush=True,
            )
            if debug and device.type == "cuda":
                dbg(f"[mem] {cuda_mem_str()}", debug)

        if args.max_cards and cards_processed >= args.max_cards:
            break

    if overall_count_gray == 0 or overall_count_rgb == 0:
        raise RuntimeError("No samples processed. Check dataset/index.")

    # -------------------------
    # Finalize overall means and whitening std
    # -------------------------
    overall_mean_rgb = (overall_sum_rgb / float(overall_count_rgb)).to(torch.float32)   # (C,)
    overall_mean_gray = (overall_sum_gray / float(overall_count_gray)).to(torch.float32)

    std_rgb = safe_std_from_sums(overall_sum_rgb, overall_sumsq_rgb, overall_count_rgb).to(torch.float32)
    std_gray = safe_std_from_sums(overall_sum_gray, overall_sumsq_gray, overall_count_gray).to(torch.float32)
    std_all = safe_std_from_sums(overall_sum_all, overall_sumsq_all, overall_count_all).to(torch.float32)

    if args.whiten_kind == "rgb":
        whiten_std = std_rgb
        whiten_name = "std_rgb"
    elif args.whiten_kind == "gray":
        whiten_std = std_gray
        whiten_name = "std_gray"
    else:
        whiten_std = std_all
        whiten_name = "std_all"

    # Filter by min_count (use rgb count as reference)
    keep_types = {t for t, c in per_type_count_rgb.items() if c >= args.min_count}

    # Decide save dtype
    save_dtype = np.float16 if args.save_dtype == "float16" else np.float32
    dtype_name = "float16" if save_dtype == np.float16 else "float32"

    # -------------------------
    # Build per-type payload
    # -------------------------
    per_type_payload: Dict[str, Any] = {}

    a = float(args.hybrid_alpha)
    a = max(0.0, min(1.0, a))

    for t in sorted(keep_types):
        mean_rgb = (per_type_sum_rgb[t] / float(per_type_count_rgb[t])).to(torch.float32)   # (C,)
        mean_gray = (per_type_sum_gray[t] / float(per_type_count_gray[t])).to(torch.float32)

        delta_mean = (mean_gray - overall_mean_gray)  # base mean delta (gray is usually less color-biased)
        delta_mean_w = delta_mean / (whiten_std + 1e-6)   # whitened

        delta_mean_w = l2_normalize(delta_mean_w)

        # pooled gradient direction (already normalized per-view; we normalize again after averaging)
        have_grad = args.use_grad_dir and per_type_count_grad.get(t, 0) > 0
        if have_grad:
            grad_dir = (per_type_sum_grad[t] / float(per_type_count_grad[t])).to(torch.float32)
            grad_dir = l2_normalize(grad_dir)
        else:
            grad_dir = torch.zeros_like(delta_mean_w)

        # hybrid direction
        if have_grad:
            delta_hybrid = l2_normalize((1.0 - a) * delta_mean_w + a * grad_dir)
        else:
            delta_hybrid = delta_mean_w

        # choose default delta
        delta_default = delta_hybrid if have_grad else delta_mean_w

        per_type_payload[t] = {
            "count_rgb_views": int(per_type_count_rgb[t]),
            "count_gray_views": int(per_type_count_gray[t]),
            "count_grad_views": int(per_type_count_grad.get(t, 0)),

            # debugging / analysis
            "mean_rgb": pack_tensor(mean_rgb, save_dtype, dtype_name),
            "mean_gray": pack_tensor(mean_gray, save_dtype, dtype_name),

            # recommended directions
            "delta_mean_w": pack_tensor(delta_mean_w, save_dtype, dtype_name),
            "delta_grad": pack_tensor(grad_dir, save_dtype, dtype_name),
            "delta_hybrid": pack_tensor(delta_hybrid, save_dtype, dtype_name),

            # what the UI should use by default
            "delta": pack_tensor(delta_default, save_dtype, dtype_name),
            "delta_kind": "hybrid" if have_grad else "mean_whitened",
            "delta_norm": float(torch.linalg.norm(delta_default).item()),
        }

    out: Dict[str, Any] = {
        "meta": {
            "index": str(Path(args.index)),
            "checkpoint": str(Path(args.model)),
            "img_size": int(cfg.img_size),
            "latent_shape": [C],  # IMPORTANT: now vectors, not maps
            "cards_processed": int(cards_processed),
            "views_rgb": int(overall_count_rgb),
            "views_gray": int(overall_count_gray),
            "min_count": int(args.min_count),
            "return_views": bool(args.return_views),

            "use_grad_dir": bool(args.use_grad_dir),
            "grad_microbatch": int(args.grad_microbatch),
            "grad_abs": bool(args.grad_abs),
            "hybrid_alpha": float(args.hybrid_alpha),

            "whiten_kind": str(args.whiten_kind),
            "whiten_std_saved_as": whiten_name,

            "note": (
                "This file stores CHANNEL-ONLY latent directions (C,) computed from pooled bottleneck z (B,C,12,12). "
                "Apply by broadcasting: z' = z + alpha * delta[:,None,None]. "
                "delta_mean_w = whitened mean-diff (gray) direction; "
                "delta_grad = pooled classifier gradient direction (optional); "
                "delta_hybrid = normalized blend; delta = default."
            ),
        },
        "overall_mean_rgb": pack_tensor(overall_mean_rgb, save_dtype, dtype_name),
        "overall_mean_gray": pack_tensor(overall_mean_gray, save_dtype, dtype_name),
        "std_rgb": pack_tensor(std_rgb, save_dtype, dtype_name),
        "std_gray": pack_tensor(std_gray, save_dtype, dtype_name),
        "std_all": pack_tensor(std_all, save_dtype, dtype_name),
        "per_type": per_type_payload,
    }

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("====================================", flush=True)
    print("Saved creature embeddings JSON:", flush=True)
    print(out_path, flush=True)
    print(f"cards_processed: {cards_processed}", flush=True)
    print(f"types saved    : {len(per_type_payload)} (min_count={args.min_count})", flush=True)
    print(f"latent shape   : {out['meta']['latent_shape']} (CHANNEL VECTORS)", flush=True)
    print("====================================", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)