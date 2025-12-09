#!/usr/bin/env python3
"""
DDPM latent→latent inference (T1n + T2F + seg  →  T1c) with tumor & whole-volume metrics.

This mirrors the rectified-flow evaluation script's behavior (I/O, metrics, CSV schema,
conditioning order, flips, normalization) but uses a classic DDPM sampler.

Key points
- Input ordering to the UNet matches training: [x_t, T1n, T2F, seg]
- Models are run in fp16 to match prior usage
- Metrics: NMSE, PSNR(dB), NCC for whole & tumor (labels 1 or 3)
- Writes per-case prediction NIfTI with seg affine and a CSV of metrics
"""

import argparse, json, warnings, csv, importlib
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast

import nibabel as nib
from monai.config import print_config
from monai.utils import set_determinism
from monai.data import NibabelWriter
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler

# -----------------------------------------------------------------------------
# Small Hydra-like instantiate (same style you use elsewhere)
# -----------------------------------------------------------------------------
def _resolve(cfg: Dict[str, Any], root: Dict[str, Any]):
    if isinstance(cfg, dict):
        return {k: _resolve(v, root) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_resolve(v, root) for v in cfg]
    if isinstance(cfg, str):
        if cfg.startswith("@"):   # reference within file
            return root[cfg[1:]]
        if cfg.startswith("$@"):  # numeric literal in root
            return root[cfg[2:]]
    return cfg

def instantiate(cfg: Dict[str, Any], root: Dict[str, Any]):
    comp = cfg.copy()
    target = comp.pop("_target_")
    kwargs = _resolve(comp, root)
    module, cls = target.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)(**kwargs)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
print_config()
ap = argparse.ArgumentParser()
# --- checkpoints & configs
ap.add_argument("--ckpt-ae",      default="autoencoder_epoch273.pt")
ap.add_argument("--ckpt-unet",    default="./rflow_t1n_t2f_DDPM/unet_latest.pt", help="DDPM UNet checkpoint")
ap.add_argument("--config-unet",  default="./config/config_train_16g.json")
# --- data roots
ap.add_argument("--latent-dir",   default="./z_latent_maps/test", help="folder containing *-t1c_z_*.pt etc.")
ap.add_argument("--seg-root",     default="/data/zeidex/BraTS_preprocessed", help="BraTS tree with <split>/<cid>/<cid>-seg.nii.gz")
# --- runtime
ap.add_argument("--out-dir",      default="./ddpm_t1n_t2f_eval")
ap.add_argument("--device",       default="cuda:0")
ap.add_argument("--batch",        type=int, default=1)
ap.add_argument("--steps",        type=int, default=1000, help="DDPM denoising steps")
ap.add_argument("--save-nifti",   default=True)
ap.add_argument("--save-latents", action="store_true")
ap.add_argument("--seed",         type=int, default=0)
ap.add_argument("--flip-x", action = "store_true", help="Flip predicted volume along the x (W) axis")
ap.add_argument("--flip-y", action="store_true", help="Flip predicted volume along the y (H) axis")
ap.add_argument("--flip-z", default = True, help="Flip predicted volume along the z (D) axis")

args = ap.parse_args()

set_determinism(args.seed)
DEV = torch.device(args.device)
OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# AutoEncoder-KL (same wiring as your rectified-flow evaluation)
# -----------------------------------------------------------------------------
print("\n>> Loading AutoEncoder-KL …")
ENV_JSON   = "./maisi/configs/environment_maisi_vae_train.json"
ARCH_JSON  = "./maisi/configs/config_maisi3d-rflow.json"
TRAIN_JSON = "./maisi/configs/config_maisi_vae_train.json"  # parity

env_args = argparse.Namespace(**json.load(open(ENV_JSON)))
arch_dict = json.load(open(ARCH_JSON))
for k, v in arch_dict.items():
    setattr(env_args, k, v)
# force num_splits=1 for inference
env_args.autoencoder_def["num_splits"] = 1

from maisi.scripts.utils import define_instance  # noqa: E402
autoencoder = define_instance(env_args, "autoencoder_def").to(DEV).half()
ae_ckpt     = torch.load(args.ckpt_ae, map_location="cpu")
if "autoencoder" in ae_ckpt:
    autoencoder.load_state_dict(ae_ckpt["autoencoder"], strict=True)
else:
    autoencoder.load_state_dict(ae_ckpt, strict=True)
autoencoder.eval(); print("✅  AutoEncoder ready (fp16)")

# -----------------------------------------------------------------------------
# UNet + DDPM scheduler. IMPORTANT: in_channels = latent_channels * 4
# -----------------------------------------------------------------------------
print("\n>> Loading DDPM UNet …")
cfg = json.load(open(args.config_unet))
latent_channels = cfg.get("latent_channels", 4)
cfg["diffusion_def"]["in_channels"] = latent_channels * 3  # [x_t, t1n, t2f, seg]
unet = instantiate(cfg["diffusion_def"], cfg).to(DEV).half()

unet_ckpt = torch.load(args.ckpt_unet, map_location="cpu")
key = "unet" if isinstance(unet_ckpt, dict) and "unet" in unet_ckpt else None
unet.load_state_dict(unet_ckpt[key] if key else unet_ckpt, strict=True)
unet.eval(); print("✅  UNet checkpoint loaded (DDPM)")

scheduler = DDPMScheduler(
    num_train_timesteps= 1000,
    schedule="scaled_linear_beta",
    beta_start=0.0005,
    beta_end=0.0195,
    clip_sample=False,
)


# -----------------------------------------------------------------------------
# Dataset – pairs latents, INCLUDING T2F to match training
# -----------------------------------------------------------------------------
class LatentQuadDataset(Dataset):
    """
    Returns dict: target=z_t1c, cond=z_t1n, cond2=z_t2f, seg=z_seg, cid, split
    """
    def __init__(self, split_dir: Path):
        self._items = []
        for mu_tgt in split_dir.rglob("*-t1c_z_mu.pt"):
            sig_tgt = mu_tgt.with_name(mu_tgt.name.replace("_z_mu.pt", "_z_sigma.pt"))
            mu_c1   = mu_tgt.with_name(mu_tgt.name.replace("-t1c_z_mu.pt", "-t1n_z_mu.pt"))
            sig_c1  = mu_c1.with_name(mu_c1.name.replace("_z_mu.pt", "_z_sigma.pt"))
            mu_c2   = mu_tgt.with_name(mu_tgt.name.replace("-t1c_z_mu.pt", "-t2f_z_mu.pt"))
            sig_c2  = mu_c2.with_name(mu_c2.name.replace("_z_mu.pt", "_z_sigma.pt"))
            if all(p.exists() for p in (mu_tgt, sig_tgt, mu_c1, sig_c1, mu_c2, sig_c2)):
                self._items.append((mu_tgt, sig_tgt, mu_c1, sig_c1, mu_c2, sig_c2))
        if not self._items:
            raise RuntimeError(f"No latent files found under {split_dir}")

    def __len__(self): return len(self._items)

    def __getitem__(self, idx):
        (mu_tgt, sig_tgt, mu_c1, sig_c1, mu_c2, sig_c2) = self._items[idx]
        μt, σt, μ1, σ1, μ2, σ2 = [torch.load(p) for p in
                                          (mu_tgt, sig_tgt, mu_c1, sig_c1, mu_c2, sig_c2)]
        z_tgt  = μt + σt * torch.randn_like(μt)
        z_c1   = μ1 + σ1 * torch.randn_like(μ1)
        z_c2   = μ2 + σ2 * torch.randn_like(μ2)
        return {
            "target": z_tgt, "cond": z_c1, "cond2": z_c2,
            "cid": mu_tgt.parent.name, "split": mu_tgt.parent.parent.name
        }

root = Path(args.latent_dir)
loader = DataLoader(
    LatentQuadDataset(root),
    batch_size=max(1, args.batch),
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------
def minmax01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # normalize per-sample (B dimension)
    dims = tuple(range(1, x.ndim))
    mn = x.amin(dim=dims, keepdim=True)
    mx = x.amax(dim=dims, keepdim=True)
    return (x - mn) / (mx - mn + eps)

def nmse(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor]=None, eps: float=1e-8) -> float:
    if mask is not None:
        m = mask.to(device=pred.device, dtype=torch.float32)
        num = torch.sum(((pred - gt) ** 2) * m, dtype=torch.float32)
        den = torch.sum((gt ** 2) * m, dtype=torch.float32)
    else:
        num = torch.sum((pred - gt) ** 2, dtype=torch.float32)
        den = torch.sum((gt ** 2), dtype=torch.float32)
    return (num / (den + eps)).item()

def psnr_db(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor]=None, eps: float=1e-12) -> float:
    if mask is not None:
        m = mask.to(device=pred.device, dtype=torch.float32)
        cnt = m.sum()
        if cnt <= 0:
            return float("nan")
        mse = torch.sum(((pred - gt) ** 2) * m, dtype=torch.float32) / cnt
    else:
        mse = torch.mean((pred - gt) ** 2, dtype=torch.float32)
    if mse <= 0:
        return float("inf")
    return (20.0 * torch.log10(torch.tensor(1.0)) - 10.0 * torch.log10(mse + eps)).item()

def ncc(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor]=None, eps: float=1e-8) -> float:
    p = pred.float()
    g = gt.float()
    if mask is not None:
        m = mask.to(device=p.device, dtype=torch.float32)
        cnt = m.sum()
        if cnt <= 0:
            return float("nan")
        pm = (p * m).sum() / cnt
        gm = (g * m).sum() / cnt
        pv = (p - pm) * m
        gv = (g - gm) * m
        denom = torch.sqrt((pv * pv).sum()) * torch.sqrt((gv * gv).sum())
        if denom < eps:
            return float("nan")
        return ((pv * gv).sum() / (denom + eps)).item()
    pm = p.mean(); gm = g.mean()
    pv = p - pm;   gv = g - gm
    denom = torch.sqrt((pv * pv).sum()) * torch.sqrt((gv * gv).sum())
    if denom < eps:
        return float("nan")
    return (pv * gv).sum().div(denom + eps).item()

def load_seg_and_affine(seg_root: Path, split: str, cid: str) -> Tuple[np.ndarray, np.ndarray]:
    cand = seg_root / split / cid / f"{cid}-seg.nii.gz"
    if not cand.exists():
        case_dir = (seg_root / split / cid)
        matches = sorted(case_dir.glob("*-seg.nii.gz"))
        if not matches:
            raise FileNotFoundError(f"No segmentation NIfTI found for {split}/{cid} under {case_dir}")
        cand = matches[0]
    img = nib.load(str(cand))
    data = img.get_fdata().astype(np.int16)
    return data, img.affine

def apply_flips(vol: torch.Tensor, flip_x: bool=False, flip_y: bool=False, flip_z: bool=False) -> torch.Tensor:
    dims = []
    if flip_z: dims.append(-3)
    if flip_y: dims.append(-2)
    if flip_x: dims.append(-1)
    return torch.flip(vol, dims=dims) if dims else vol

# -----------------------------------------------------------------------------
# CSV header
# -----------------------------------------------------------------------------
csv_path = OUT / "metrics_ddpm_t1n_t2f_seg.csv"
write_header = not csv_path.exists()
csv_f = open(csv_path, "a", newline="")
csv_w = csv.writer(csv_f)
if write_header:
    csv_w.writerow([
        "cid","split",
        "NMSE_whole","PSNR_whole_dB","NCC_whole",
        "NMSE_tumor","PSNR_tumor_dB","NCC_tumor",
        "pred_nii_path","pred_latent_path"
    ])

# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
print("\n>> Inference (DDPM) …")
seg_root = Path(args.seg_root)

with torch.no_grad():
    for batch in DataLoader(LatentQuadDataset(Path(args.latent_dir)), batch_size=max(1, args.batch), shuffle=False, num_workers=2, pin_memory=True):
        z_cond  = batch["cond"].to(DEV)   # T1n latent
        z_cond2 = batch["cond2"].to(DEV)  # T2F latent
        z_tgt   = batch["target"].to(DEV) # T1c latent (for decoding GT)

        B = z_cond.size(0)
        z_curr = torch.randn_like(z_cond)  # start from pure noise

        # Prepare scheduler timesteps
        scheduler.set_timesteps(args.steps)
        timesteps = scheduler.timesteps
        timesteps = timesteps.to(device=DEV, dtype=torch.long)

        # Denoising loop
        with autocast(device_type=DEV.type, dtype=torch.float16):
            for t in timesteps:
                t_b = t.expand(B)
                unet_in = torch.cat([z_curr, z_cond, z_cond2], dim=1)
                noise_pred = unet(unet_in, t_b)

                out = scheduler.step(noise_pred, t, z_curr)
                if isinstance(out, tuple):
                    z_curr = out[0]
                elif hasattr(out, "prev_sample"):
                    z_curr = out.prev_sample
                elif hasattr(out, "sample"):
                    z_curr = out.sample
                else:
                    z_curr = out

        z_pred = z_curr

        # Decode to image space
        # Autoencoder.decode expects half→float for safety
        img_pred = autoencoder.decode(z_pred).float().cpu()  # B,1,D,H,W
        img_gt   = autoencoder.decode(z_tgt).float().cpu()

        img_pred = apply_flips(img_pred, args.flip_x, args.flip_y, args.flip_z)
        img_gt   = apply_flips(img_gt, args.flip_x, args.flip_y, args.flip_z)

        # Normalize to [0,1] per volume before metrics
        def minmax01(x, eps: float = 1e-8):
            dims = tuple(range(1, x.ndim))
            mn = x.amin(dim=dims, keepdim=True)
            mx = x.amax(dim=dims, keepdim=True)
            return (x - mn) / (mx - mn + eps)

        img_pred_n = minmax01(img_pred)
        img_gt_n   = minmax01(img_gt)

        # Per-case bookkeeping
        for i in range(B):
            cid   = batch["cid"][i]
            split = batch["split"][i]
            case_dir = OUT / split / cid
            case_dir.mkdir(parents=True, exist_ok=True)

            # Load GT seg (for mask + affine)
            seg_np, affine = load_seg_and_affine(seg_root, split, cid)

            # Ensure shapes align (D,H,W)
            vol_pred = img_pred_n[i, 0]
            vol_gt   = img_gt_n[i, 0]

            if tuple(seg_np.shape) != tuple(vol_pred.shape):
                warnings.warn(f"Shape mismatch for {split}/{cid}: seg {seg_np.shape} vs img {tuple(vol_pred.shape)}")
                if np.prod(seg_np.shape) != np.prod(vol_pred.shape):
                    raise RuntimeError(f"Incompatible shapes for metrics in {split}/{cid}")

            # Tumor mask: labels 1 or 3
            tumor_mask = torch.from_numpy((seg_np == 1) | (seg_np == 3))

            # Metrics
            nmse_whole = nmse(vol_pred, vol_gt, None)
            # PSNR assumes [0,1]; already normalized
            mse_whole = torch.mean((vol_pred - vol_gt) ** 2, dtype=torch.float32)
            psnr_whole = (20.0 * torch.log10(torch.tensor(1.0)) - 10.0 * torch.log10(mse_whole + 1e-12)).item()
            pm = vol_pred.mean(); gm = vol_gt.mean()
            pv = (vol_pred - pm);  gv = (vol_gt - gm)
            denom = torch.sqrt((pv * pv).sum()) * torch.sqrt((gv * gv).sum())
            ncc_whole = ((pv * gv).sum() / (denom + 1e-8)).item() if denom > 0 else float("nan")

            nmse_tumor = float("nan"); psnr_tumor = float("nan"); ncc_tumor = float("nan")
            if tumor_mask.any():
                nmse_tumor = nmse(vol_pred, vol_gt, tumor_mask)
                cnt = tumor_mask.sum()
                mse_tumor = ( ((vol_pred - vol_gt) ** 2) * tumor_mask.to(vol_pred.dtype) ).sum() / cnt
                psnr_tumor = (20.0 * torch.log10(torch.tensor(1.0)) - 10.0 * torch.log10(mse_tumor + 1e-12)).item()
                pm = (vol_pred * tumor_mask).sum() / cnt
                gm = (vol_gt   * tumor_mask).sum() / cnt
                pv = (vol_pred - pm) * tumor_mask
                gv = (vol_gt   - gm) * tumor_mask
                denom = torch.sqrt((pv * pv).sum()) * torch.sqrt((gv * gv).sum())
                ncc_tumor = ((pv * gv).sum() / (denom + 1e-8)).item() if denom > 0 else float("nan")

            print(
                f"[{split}/{cid}] "
                f"Whole → NMSE={nmse_whole:.6f}, PSNR={psnr_whole:.2f} dB, NCC={ncc_whole:.4f} | "
                f"Tumor → NMSE={nmse_tumor:.6f}, PSNR={psnr_tumor:.2f} dB, NCC={ncc_tumor:.4f}"
            )

            # Save outputs
            lat_path = ""
            if args.save_latents:
                lat_path = str(case_dir / f"{cid}_t1c_pred_latent.pt")
                torch.save(z_pred[i].cpu(), lat_path)

            nii_path = ""
            if args.save_nifti:
                nii_path = str(case_dir / f"{cid}_t1c_pred.nii.gz")
                arr = img_pred[i].cpu().numpy()  # 1,D,H,W
                writer = NibabelWriter()
                writer.set_data_array(arr, channel_dim=0)
                writer.set_metadata({"affine": affine, "original_affine": affine})
                writer.write(nii_path, verbose=False)

            # CSV row
            csv_w.writerow([
                cid, split,
                f"{nmse_whole:.6f}", f"{psnr_whole:.3f}", f"{ncc_whole:.5f}",
                f"{nmse_tumor:.6f}", f"{psnr_tumor:.3f}", f"{ncc_tumor:.5f}",
                nii_path, lat_path
            ])
            csv_f.flush()

csv_f.close()
print("\n✅ Done. Metrics saved to:", csv_path)
print("   Output dir:", OUT.resolve())
