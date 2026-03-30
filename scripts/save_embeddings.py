# scripts/save_embeddings.py
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


# -------------------------
# Helpers
# -------------------------
def encode_array_to_b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


def pack_tensor(t: torch.Tensor, dtype=np.float16):
    arr = t.detach().cpu().numpy().astype(dtype, copy=False)
    return {
        "dtype": "float16" if dtype == np.float16 else "float32",
        "shape": list(arr.shape),
        "b64": encode_array_to_b64(arr),
    }

def simple_collate(batch):
    return {
        "x_in": torch.stack([b["x_in"] for b in batch], dim=0),
        "creature_types": [b.get("creature_types", []) for b in batch],
    }

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_type_vocab(index_path: Path) -> List[str]:
    types = set()
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            for t in e.get("creature_types", []):
                if isinstance(t, str) and t:
                    types.add(t)
    return sorted(types)


# -------------------------
# Main
# -------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("Simple embedding extractor")
    ap.add_argument("--index", default="data/index.jsonl")
    ap.add_argument("--model", default="models/model.pt")
    ap.add_argument("--out", default="data/creature_embeddings.json")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--min_count", type=int, default=20)

    args = ap.parse_args()

    root = project_root()
    index_path = root / args.index
    model_path = root / args.model
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    )

    print("=== Simple Embedding Extraction ===")
    print("device:", device)

    # -------------------------
    # Load model
    # -------------------------
    from create_autoencoder_model import AEConfig, Autoencoder
    from dataloader import MTGCreatureArtDataset

    payload = torch.load(model_path, map_location="cpu")
    cfg = AEConfig(**payload["config"])

    model = Autoencoder(cfg).to(device)
    model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()

    C = cfg.bottleneck_channels

    # -------------------------
    # Dataset
    # -------------------------
    ds = MTGCreatureArtDataset(
        index_path=index_path,
        img_size=cfg.img_size,
        flip_p=0.0,
        denoise=False,
    )

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=simple_collate,
    )

    # -------------------------
    # Accumulators
    # -------------------------
    global_sum = torch.zeros(C, dtype=torch.float64)
    global_count = 0

    per_type_sum: Dict[str, torch.Tensor] = {}
    per_type_count: Dict[str, int] = {}

    # -------------------------
    # Loop
    # -------------------------
    for step, batch in enumerate(dl):
        x = batch["x_in"].to(device)  # (B,3,H,W)
        types_list = batch["creature_types"]

        z = model.encoder(x)               # (B,C,H,W)
        v = z.mean(dim=(2, 3))             # (B,C)

        v_cpu = v.detach().cpu().double()

        global_sum += v_cpu.sum(dim=0)
        global_count += v_cpu.shape[0]

        for i in range(v_cpu.shape[0]):
            vec = v_cpu[i]
            for t in types_list[i] or []:
                if not t:
                    continue
                if t not in per_type_sum:
                    per_type_sum[t] = torch.zeros(C, dtype=torch.float64)
                    per_type_count[t] = 0

                per_type_sum[t] += vec
                per_type_count[t] += 1

        if step % 50 == 0:
            print(f"step {step} | processed {global_count}")

    # -------------------------
    # Finalize
    # -------------------------
    global_mean = (global_sum / global_count).float()

    per_type_payload = {}

    for t, count in per_type_count.items():
        if count < args.min_count:
            continue

        mean_t = (per_type_sum[t] / count).float()
        delta = mean_t - global_mean

        per_type_payload[t] = {
            "count": int(count),
            "mean": pack_tensor(mean_t),
            "delta": pack_tensor(delta),
        }

    # -------------------------
    # Save
    # -------------------------
    out = {
        "meta": {
            "latent_dim": C,
            "cards_processed": int(global_count),
            "min_count": args.min_count,
            "note": "Simple mean-based embeddings: delta = mean(type) - global_mean",
        },
        "global_mean": pack_tensor(global_mean),
        "per_type": per_type_payload,
    }

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=== DONE ===")
    print("saved to:", out_path)
    print("types:", len(per_type_payload))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)