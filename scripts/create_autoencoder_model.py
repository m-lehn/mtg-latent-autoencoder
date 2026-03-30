import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import argparse

import torch
import torch.nn as nn

# -----------------------------
# Config
# -----------------------------
@dataclass
class AEConfig:
    img_size: int = 384
    in_channels: int = 3

    base_channels: int = 64
    channel_mults: tuple = (1, 2, 4, 8, 8)   # 5 downsamples -> 12x12

    bottleneck_channels: int = 256           # z shape: (B, 256, 12, 12)

    # Decoder refinement depth (most useful at high res)
    refine_192: int = 2                      # #ResBlocks after 96->192 up
    refine_384: int = 4                      # #ResBlocks after 192->384 up

    use_sigmoid: bool = True

    # --------- NEW: latent classifier head (optional) ----------
    # If num_types is None or 0, no classifier head is created.
    num_types: int | None = None             # set automatically from index.jsonl if provided
    type_head_hidden: int = 256              # MLP hidden size
    type_head_dropout: float = 0.10          # dropout in head

# -----------------------------
# count creature types for classifier
# -----------------------------
def count_creature_types_from_jsonl(index_path: Path) -> int:
    """
    Counts unique creature type strings in a JSONL index.
    Each line is a dict that should contain "creature_types": [..].
    """
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    types: set[str] = set()
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
        raise RuntimeError(f"No creature_types found in: {index_path}")

    return len(types)

# -----------------------------
# Blocks
# -----------------------------
class ResBlock(nn.Module):
    """Simple residual block: (conv-relu-conv) + skip."""
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class DownBlock(nn.Module):
    """ResBlock(s) then stride-2 downsample conv."""
    def __init__(self, c_in: int, c_out: int, n_blocks: int = 1):
        super().__init__()
        blocks = [nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True)]
        blocks += [ResBlock(c_out) for _ in range(n_blocks)]
        self.pre = nn.Sequential(*blocks)
        self.down = nn.Conv2d(c_out, c_out, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.pre(x)
        x = self.down(x)
        return x


class PixelShuffleUp(nn.Module):
    """Conv -> PixelShuffle(2) -> ReLU."""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    """Upsample then ResBlock refinement."""
    def __init__(self, c_in: int, c_out: int, n_refine: int = 1):
        super().__init__()
        self.up = PixelShuffleUp(c_in, c_out)
        self.refine = nn.Sequential(*[ResBlock(c_out) for _ in range(n_refine)])

    def forward(self, x):
        x = self.up(x)
        x = self.refine(x)
        return x


# -----------------------------
# Latent classifier head
# -----------------------------
class LatentTypeHead(nn.Module):
    """
    Multi-label classifier from latent map z (B,C,H,W).
    Spatial conv head:
      - mixes channels early (1x1 conv)
      - uses spatial patterns (3x3 conv)
      - pools only at the end
    """
    def __init__(self, c_in: int, num_types: int, hidden: int = 256, dropout: float = 0.10):
        super().__init__()
        # "hidden" here is the conv width (keeps your config name stable)
        c_mid = int(hidden)

        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_mid, kernel_size=1),      # channel mixing
            nn.GELU(),
            nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),                    # (B, c_mid, 1, 1)
            nn.Flatten(),                               # (B, c_mid)
            nn.Linear(c_mid, num_types),                # logits
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

# -----------------------------
# Encoder / Decoder
# -----------------------------
class Encoder(nn.Module):
    """
    384 -> 192 -> 96 -> 48 -> 24 -> 12 (5 downsamples)
    Output z: (B, bottleneck_channels, 12, 12)
    """
    def __init__(self, cfg: AEConfig):
        super().__init__()
        c0 = cfg.base_channels
        chs = [c0 * m for m in cfg.channel_mults]  # e.g. [64,128,256,512,512]

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, chs[0], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.down1 = DownBlock(chs[0], chs[0], n_blocks=1)  # 384 -> 192
        self.down2 = DownBlock(chs[0], chs[1], n_blocks=1)  # 192 -> 96
        self.down3 = DownBlock(chs[1], chs[2], n_blocks=1)  # 96  -> 48
        self.down4 = DownBlock(chs[2], chs[3], n_blocks=1)  # 48  -> 24
        self.down5 = DownBlock(chs[3], chs[4], n_blocks=1)  # 24  -> 12

        self.to_bn = nn.Conv2d(chs[4], cfg.bottleneck_channels, kernel_size=1)
        self.bn_refine = nn.Sequential(
            ResBlock(cfg.bottleneck_channels),
            ResBlock(cfg.bottleneck_channels),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        z = self.to_bn(x)
        z = self.bn_refine(z)
        return z


class Decoder(nn.Module):
    """
    (B, bnC, 12, 12) -> 24 -> 48 -> 96 -> 192 -> 384 -> output
    Single output only.
    """
    def __init__(self, cfg: AEConfig):
        super().__init__()
        c0 = cfg.base_channels
        chs = [c0 * m for m in cfg.channel_mults]  # [64,128,256,512,512]

        self.from_bn = nn.Conv2d(cfg.bottleneck_channels, chs[4], kernel_size=1)

        self.up1 = UpBlock(chs[4], chs[3], n_refine=1)               # 12 -> 24
        self.up2 = UpBlock(chs[3], chs[2], n_refine=1)               # 24 -> 48
        self.up3 = UpBlock(chs[2], chs[1], n_refine=1)               # 48 -> 96
        self.up4 = UpBlock(chs[1], chs[0], n_refine=cfg.refine_192)   # 96 -> 192

        # 192 -> 384, keep channels and spend refinement here
        self.up5_up = PixelShuffleUp(chs[0], chs[0])
        self.up5_refine = nn.Sequential(*[ResBlock(chs[0]) for _ in range(cfg.refine_384)])

        # Output head
        head = [nn.Conv2d(chs[0], cfg.in_channels, kernel_size=3, padding=1)]
        if cfg.use_sigmoid:
            head.append(nn.Sigmoid())
        self.head = nn.Sequential(*head)

    def forward(self, z):
        x = self.from_bn(z)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5_up(x)
        x = self.up5_refine(x)
        y = self.head(x)
        return y

class TypeHead(nn.Module): # Old
    """
    Maps bottleneck z (B, C, 12, 12) -> logits (B, num_types)
    Uses global average pooling so it doesn't explode params.
    """
    def __init__(self, in_channels: int, num_types: int, hidden: int = 256, dropout: float = 0.10):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 1, 1)
        self.net = nn.Sequential(
            nn.Flatten(),                    # (B, C)
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_types),    # logits
        )

    def forward(self, z):
        z = self.pool(z)
        return self.net(z)

class Autoencoder(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

        if cfg.num_types is None or cfg.num_types <= 0:
            raise ValueError(
                "cfg.num_types must be a positive int for the fixed autoencoder+classifier setup."
            )

        self.type_head = LatentTypeHead(
            c_in=cfg.bottleneck_channels,
            num_types=cfg.num_types,
            hidden=cfg.type_head_hidden,
            dropout=cfg.type_head_dropout,
        )

    def forward(self, x):
        """
        Always returns:
          y      : reconstructed image, (B, 3, H, W)
          logits : creature-type logits, (B, num_types)
        """
        z = self.encoder(x)          # (B, bottleneck_channels, 12, 12)
        y = self.decoder(z)          # (B, 3, 384, 384)
        logits = self.type_head(z)   # (B, num_types)
        return y, logits


# -----------------------------
# Save untrained model
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default="data/index.jsonl",
                    help="Path to index.jsonl used to auto-count creature types.")
    ap.add_argument("--auto_num_types", action="store_true",
                    help="If set, compute num_types from --index and enable classifier head.")
    ap.add_argument("--num_types", type=int, default=0,
                    help="Manually set num_types (overrides auto if >0). Use 0 to disable.")
    args = ap.parse_args()

    num_types = None
    if args.num_types and args.num_types > 0:
        num_types = args.num_types
    elif args.auto_num_types:
        num_types = count_creature_types_from_jsonl(Path(args.index))

    cfg = AEConfig(
        img_size=384,
        base_channels=64,
        channel_mults=(1, 2, 4, 8, 8),
        bottleneck_channels=256,
        refine_192=2,
        refine_384=4,
        use_sigmoid=True,

        # classifier head
        num_types=num_types,
        type_head_hidden=256,
        type_head_dropout=0.10,
    )

    model = Autoencoder(cfg).eval()

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_path = out_dir / "model.pt"
    payload = {"config": asdict(cfg), "state_dict": model.state_dict()}
    torch.save(payload, pt_path)
    print(f"Saved untrained model to: {pt_path}")

    json_path = out_dir / "model.json"
    json_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(f"Saved config to: {json_path}")

    if cfg.num_types is not None:
        print(f"num_types (classifier head) = {cfg.num_types}")

if __name__ == "__main__":
    main()
