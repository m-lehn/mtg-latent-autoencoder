from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Basic losses (thin wrappers)
# -------------------------
class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.fn(pred, target)


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.fn(pred, target)


# -------------------------
# Charbonnier (smooth L1-like)
# -------------------------
class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss: mean(sqrt((x-y)^2 + eps^2))
    Often used instead of L1 for stable gradients and less "washed out" results.
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + (self.eps * self.eps)))


# -------------------------
# Edge loss (Sobel gradients)
# -------------------------
def _sobel_kernels(device, dtype):
    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.],
                       [ 0.,  0.,  0.],
                       [ 1.,  2.,  1.]], device=device, dtype=dtype).view(1, 1, 3, 3)
    return kx, ky


def _rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,3,H,W) -> (B,1,H,W)
    """
    if x.size(1) == 1:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


class EdgeLoss(nn.Module):
    """
    Edge/gradient loss using Sobel magnitude.
    Compares gradient magnitude maps between pred and target.
    """
    def __init__(self, use_luma: bool = True, loss: str = "l1"):
        super().__init__()
        self.use_luma = use_luma
        self.loss = loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_g = _rgb_to_luma(pred.float()) if self.use_luma else pred.float()
        targ_g = _rgb_to_luma(target.float()) if self.use_luma else target.float()

        b, c, h, w = pred_g.shape
        kx, ky = _sobel_kernels(pred_g.device, pred_g.dtype)

        # depthwise conv for each channel
        kx = kx.repeat(c, 1, 1, 1)
        ky = ky.repeat(c, 1, 1, 1)

        gx_p = F.conv2d(pred_g, kx, padding=1, groups=c)
        gy_p = F.conv2d(pred_g, ky, padding=1, groups=c)
        gx_t = F.conv2d(targ_g, kx, padding=1, groups=c)
        gy_t = F.conv2d(targ_g, ky, padding=1, groups=c)

        mag_p = torch.sqrt(gx_p * gx_p + gy_p * gy_p + 1e-12)
        mag_t = torch.sqrt(gx_t * gx_t + gy_t * gy_t + 1e-12)

        if self.loss == "mse":
            return F.mse_loss(mag_p, mag_t)
        return F.l1_loss(mag_p, mag_t)


# -------------------------
# Total Variation (optional regularizer)
# -------------------------
class TotalVariationLoss(nn.Module):
    """
    Encourages spatial smoothness. Use tiny weight if at all.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return dh + dw


# -------------------------
# Perceptual loss (VGG16 features)
# -------------------------
class PerceptualLossVGG(nn.Module):
    """
    VGG16 perceptual loss:
      - expects input in [0,1] RGB
      - internally applies ImageNet normalization
    """
    def __init__(
        self,
        layers: Tuple[str, ...] = ("relu2_2", "relu3_3"),
        layer_weights: Optional[Tuple[float, ...]] = None,
        use_l1: bool = True,
    ):
        super().__init__()
        self.layers = layers
        self.layer_weights = layer_weights or tuple([1.0] * len(layers))
        if len(self.layer_weights) != len(self.layers):
            raise ValueError("layer_weights must match layers length")

        self.use_l1 = use_l1

        # Lazy import
        from torchvision.models import vgg16, VGG16_Weights

        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

        self.name_to_idx = {
            "relu1_2": 3,
            "relu2_2": 8,
            "relu3_3": 15,
            "relu4_3": 22,
            "relu5_3": 29,
        }
        missing = [n for n in self.layers if n not in self.name_to_idx]
        if missing:
            raise ValueError(f"Unknown VGG layer names: {missing}")

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def _extract(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: Dict[str, torch.Tensor] = {}
        x = self._norm(x)

        max_idx = max(self.name_to_idx[n] for n in self.layers)
        for i, m in enumerate(self.vgg):
            x = m(x)
            for name, idx in self.name_to_idx.items():
                if i == idx and name in self.layers:
                    feats[name] = x
            if i >= max_idx:
                break
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fp = self._extract(pred)
        ft = self._extract(target)

        total = 0.0
        for (name, w) in zip(self.layers, self.layer_weights):
            if self.use_l1:
                total = total + w * F.l1_loss(fp[name], ft[name])
            else:
                total = total + w * F.mse_loss(fp[name], ft[name])
        return total


# -------------------------
# Multi-label creature-type classification loss
# -------------------------
class CreatureTypeBCELoss(nn.Module):
    """
    Multi-label BCE loss for creature-type logits.

    - logits: (B, num_types) raw (no sigmoid)
    - targets: (B, num_types) float in {0,1} (multi-hot)

    Optional:
      - label_smoothing: tiny smoothing can help stability
      - pos_weight: tensor (num_types,) to upweight rare positives
    """
    def __init__(
        self,
        label_smoothing: float = 0.0,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("pos_weight", pos_weight, persistent=False)
        self._loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or targets.ndim != 2:
            raise ValueError(f"logits/targets must be 2D (B,C). Got {logits.shape=} {targets.shape=}")
        if logits.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {logits.shape=} vs {targets.shape=}")

        t = targets
        if self.label_smoothing > 0:
            # Smooth only slightly toward 0.5
            eps = self.label_smoothing
            t = t * (1.0 - eps) + 0.5 * eps

        return self._loss(logits, t)


# -------------------------
# PatchCritic degrade-loss (learned perceptual-ish penalty)
# -------------------------
class PatchCriticDegradeLoss(nn.Module):
    """
    Uses a trained PatchCritic that outputs logits map (B,1,h,w).
    We interpret sigmoid(logits) as "degradedness" probability:
      clean ~= 0, degraded ~= 1

    Loss returned is mean over all patches and batch:
      loss = mean(sigmoid(logits))

    IMPORTANT:
      - Critic parameters are frozen (no grads for critic weights)
      - We do NOT use torch.no_grad() in forward, so gradients flow into pred.
    """
    _cached_model: Optional[nn.Module] = None
    _cached_device: Optional[torch.device] = None
    _cached_path: Optional[str] = None

    def __init__(
        self,
        ckpt_path: str = "models/patch_critic.pt",
        script_module: str = "create_train_patch_critic",  # scripts/create_train_patch_critic.py
        map_reduce: str = "mean",  # mean or sum
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.script_module = script_module
        self.map_reduce = map_reduce
        self.use_sigmoid = use_sigmoid

    @staticmethod
    def _project_root() -> Path:
        # scripts/loss.py -> scripts/ -> project root
        return Path(__file__).resolve().parents[1]

    def _load_critic(self, device: torch.device) -> nn.Module:
        # Cache by (path, device)
        abs_path = Path(self.ckpt_path)
        if not abs_path.is_absolute():
            abs_path = (self._project_root() / abs_path).resolve()

        cache_hit = (
            self._cached_model is not None
            and self._cached_device == device
            and self._cached_path == str(abs_path)
        )
        if cache_hit:
            return self._cached_model  # type: ignore[return-value]

        # Ensure scripts/ is importable
        scripts_dir = self._project_root() / "scripts"
        import sys
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        # Import PatchCritic architecture from your training script
        mod = __import__(self.script_module)
        if not hasattr(mod, "PatchCritic"):
            raise ImportError(
                f"Module '{self.script_module}' does not expose PatchCritic. "
                f"Export it (class PatchCritic ...) in scripts/{self.script_module}.py"
            )
        PatchCritic = getattr(mod, "PatchCritic")

        model = PatchCritic().to(device)
        payload = torch.load(str(abs_path), map_location="cpu", weights_only=False)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        model.load_state_dict(state, strict=True)

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # Cache
        self.__class__._cached_model = model
        self.__class__._cached_device = device
        self.__class__._cached_path = str(abs_path)
        return model

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        pred: (B,3,H,W) in [0,1]
        returns: scalar loss
        """
        device = pred.device
        critic = self._load_critic(device)

        logits = critic(pred) # global/local/art
        #for key, value in logits.items():
        #    print(f"{key}: {type(value).__name__}")
        score_global = torch.sigmoid(logits["global"]).mean() if self.use_sigmoid else logits
        score_local = torch.sigmoid(logits["local"]).mean() if self.use_sigmoid else logits
        score_art = torch.sigmoid(logits["art"]).mean() if self.use_sigmoid else logits
        x = (score_global + score_local + score_art)*20

        if self.map_reduce == "sum":
            return x.sum() / x.numel()
        return x.mean()


# -------------------------
# Combined loss helper
# -------------------------
@dataclass
class LossWeights:
    pixel: float = 1.0
    perceptual: float = 0.0
    edge: float = 0.0
    tv: float = 0.0
    patch_critic: float = 0.0
    cls: float = 0.0 


class CombinedLoss(nn.Module):
    def __init__(
        self,
        pixel_loss: nn.Module,
        weights: LossWeights,
        perceptual_loss: Optional[nn.Module] = None,
        edge_loss: Optional[nn.Module] = None,
        tv_loss: Optional[nn.Module] = None,
        patch_critic_loss: Optional[nn.Module] = None,
        cls_loss: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.pixel_loss = pixel_loss
        self.perceptual_loss = perceptual_loss
        self.edge_loss = edge_loss
        self.tv_loss = tv_loss
        self.patch_critic_loss = patch_critic_loss
        self.cls = cls_loss
        self.w = weights

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        cls_logits: Optional[torch.Tensor] = None,
        cls_target: Optional[torch.Tensor] = None,
    ):
        comps: Dict[str, float] = {}

        lp = self.pixel_loss(pred, target)
        loss = self.w.pixel * lp
        comps["pixel"] = float(lp.detach().item())

        if self.perceptual_loss is not None and self.w.perceptual > 0:
            lperc = self.perceptual_loss(pred, target)
            loss = loss + self.w.perceptual * lperc
            comps["perceptual"] = float(lperc.detach().item())

        if self.edge_loss is not None and self.w.edge > 0:
            ledge = self.edge_loss(pred, target)
            loss = loss + self.w.edge * ledge
            comps["edge"] = float(ledge.detach().item())

        if self.tv_loss is not None and self.w.tv > 0:
            ltv = self.tv_loss(pred)
            loss = loss + self.w.tv * ltv
            comps["tv"] = float(ltv.detach().item())

        if self.patch_critic_loss is not None and self.w.patch_critic > 0:
            lpc = self.patch_critic_loss(pred)
            loss = loss + self.w.patch_critic * lpc
            comps["patch_critic"] = float(lpc.detach().item())

        # classifier term
        if self.cls is not None and self.w.cls > 0:
            if cls_logits is None or cls_target is None:
                raise ValueError("classifier loss enabled but cls_logits/cls_target not provided")
            lcls = self.cls_loss(cls_logits, cls_target)
            loss = loss + self.w.cls * lcls
            comps["cls"] = float(lcls.detach().item())

        comps["total"] = float(loss.detach().item())
        return loss, comps


# -------------------------
# Standalone test
# -------------------------
def _tstats(x: torch.Tensor) -> str:
    return (
        f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
        f"min={x.min().item():.4f} max={x.max().item():.4f} "
        f"mean={x.mean().item():.4f} std={x.std().item():.4f} "
        f"finite={torch.isfinite(x).all().item()}"
    )


def main():
    ap = argparse.ArgumentParser(description="Sanity-check loss functions on random tensors.")
    ap.add_argument("--bs", type=int, default=4, help="batch size")
    ap.add_argument("--size", type=int, default=384, help="H=W size")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--perceptual", action="store_true", help="also run VGG perceptual loss (requires torchvision)")
    ap.add_argument("--patch_critic", action="store_true", help="also run PatchCritic degrade loss if checkpoint exists")
    ap.add_argument("--patch_ckpt", type=str, default="models/patch_critic.pt")
    args = ap.parse_args()

    device = torch.device(args.device)

    # random tensors in [0,1]
    pred = torch.rand(args.bs, 3, args.size, args.size, device=device)
    target = torch.rand(args.bs, 3, args.size, args.size, device=device)

    print("=== loss.py sanity check ===")
    print("pred  :", _tstats(pred))
    print("tgt   :", _tstats(target))
    print("============================")

    l1 = L1Loss().to(device)
    mse = MSELoss().to(device)
    charbon = CharbonnierLoss(eps=1e-3).to(device)
    edge = EdgeLoss(use_luma=True, loss="l1").to(device)
    tv = TotalVariationLoss().to(device)

    with torch.no_grad():
        v_l1 = l1(pred, target).item()
        v_mse = mse(pred, target).item()
        v_char = charbon(pred, target).item()
        v_edge = edge(pred, target).item()
        v_tv = tv(pred).item()

    print(f"L1           : {v_l1:.6f}")
    print(f"MSE          : {v_mse:.6f}")
    print(f"Charbonnier  : {v_char:.6f}")
    print(f"Edge (Sobel) : {v_edge:.6f}")
    print(f"TV(pred)     : {v_tv:.6f}")

    if args.perceptual:
        try:
            perc = PerceptualLossVGG(layers=("relu2_2", "relu3_3"), use_l1=True).to(device)
            with torch.no_grad():
                v_perc = perc(pred, target).item()
            print(f"Perceptual(VGG): {v_perc:.6f}")
        except Exception as e:
            print("Perceptual(VGG): FAILED")
            print("Reason:", e)

    if args.patch_critic:
        ckpt = Path(args.patch_ckpt)
        if not ckpt.is_absolute():
            ckpt = (Path(__file__).resolve().parents[1] / ckpt).resolve()
        if ckpt.exists():
            try:
                pc = PatchCriticDegradeLoss(ckpt_path=str(ckpt)).to(device)
                # don't use torch.no_grad here if you want to test grads;
                # for a numeric sanity check it's fine:
                with torch.no_grad():
                    v_pc = pc(pred).item()
                print(f"PatchCritic(degrade) : {v_pc:.6f}  (0=clean, 1=degraded)")
            except Exception as e:
                print("PatchCritic(degrade): FAILED")
                print("Reason:", e)
        else:
            print(f"PatchCritic(degrade): skipped (checkpoint not found: {ckpt})")

    # Combined loss example
    weights = LossWeights(pixel=1.0, perceptual=0.1 if args.perceptual else 0.0, edge=0.05, tv=0.0, patch_critic=0.0)
    perceptual_loss = None
    if args.perceptual:
        try:
            perceptual_loss = PerceptualLossVGG(layers=("relu2_2", "relu3_3"), use_l1=True).to(device)
        except Exception:
            perceptual_loss = None

    combo = CombinedLoss(
        pixel_loss=CharbonnierLoss(eps=1e-3),
        weights=weights,
        perceptual_loss=perceptual_loss,
        edge_loss=edge,
        tv_loss=None,
        patch_critic_loss=None,
    ).to(device)

    with torch.no_grad():
        total, comps = combo(pred, target)

    print("\nCombinedLoss components:")
    for k, v in comps.items():
        print(f"  {k:12s}: {v:.6f}")

    print("Done.")


if __name__ == "__main__":
    main()
