#!/usr/bin/env python3
"""
R-Flow latent→latent inference (T1n + T2F  →  T1c) with whole-volume metrics only.
- Matches the training UNet input ordering: [noisy, T1n, T2F] (in_channels = latent_channels * 3)
- Decodes with the same AutoEncoder-KL
- Writes prediction NIfTI using the affine from the **T1c** file in Brats_preprocessed (not the seg)
- Metrics per-case to CSV: NMSE, PSNR(dB), NCC, SSIM (whole-volume)
"""

import argparse, json, warnings, csv, importlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from tqdm.auto import tqdm

import nibabel as nib
from monai.config import print_config
from monai.utils import set_determinism
from monai.data import NibabelWriter
from monai.metrics import PSNRMetric, SSIMMetric
from monai.networks.schedulers import RFlowScheduler

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
ap.add_argument("--ckpt_ae",      default="autoencoder_epoch273.pt")
ap.add_argument("--ckpt_unet",    default="./rflow_t1_t2f/unet_latest.pt")
ap.add_argument("--config_unet",  default="./config/config_train_16g.json")
# --- data roots
ap.add_argument("--latent_dir",   default="./z_latent_maps/test", help="folder containing *-t1c_z_*.pt etc.")
ap.add_argument("--brats_root",   default="./Brats_preprocessed", help="BraTS tree with <split>/<cid>/<cid>-t1c.nii.gz")
# --- runtime
ap.add_argument("--out_dir",      default="./rflow_t1n_t2f_eval_ablation_t2_only")
ap.add_argument("--device",       default="cuda:0")
ap.add_argument("--batch",        type=int, default=1)
ap.add_argument("--steps",        type=int, default=200)
ap.add_argument("--save_nifti",   default=True)
ap.add_argument("--save_latents", action="store_true")
ap.add_argument("--seed",         type=int, default=0)
ap.add_argument("--flip_x", action="store_true", help="Flip predicted volume along the x (W) axis")
ap.add_argument("--flip_y", action="store_true", help="Flip predicted volume along the y (H) axis")
ap.add_argument("--flip_z", default=True, help="Flip predicted volume along the z (D) axis")

args = ap.parse_args()

set_determinism(args.seed)
DEV = torch.device(args.device)
OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# AutoEncoder-KL (same as your prior test script wiring)
# -----------------------------------------------------------------------------
print("\n>> Loading AutoEncoder-KL …")
ENV_JSON   = "./maisi/configs/environment_maisi_vae_train.json"
ARCH_JSON  = "./maisi/configs/config_maisi3d-rflow.json"
TRAIN_JSON = "./maisi/configs/config_maisi_vae_train.json"  # parity

env_args = argparse.Namespace(**json.load(open(ENV_JSON)))
arch_dict = json.load(open(ARCH_JSON))
for k, v in arch_dict.items():
    setattr(env_args, k, v)
# force num_splits=1 for inference (matches your usage)
env_args.autoencoder_def["num_splits"] = 1

# define_instance exactly as in generate_latent_maps (already in your repo)
from maisi.scripts.utils import define_instance  # noqa: E402
autoencoder = define_instance(env_args, "autoencoder_def").to(DEV).half()
ae_ckpt     = torch.load(args.ckpt_ae, map_location="cpu")
autoencoder.load_state_dict(ae_ckpt.get("autoencoder", ae_ckpt), strict=True)
autoencoder.float().eval(); print("✅  AutoEncoder ready (fp16)")
ae_dev  = next(autoencoder.parameters()).device
ae_dt   = next(autoencoder.parameters()).dtype  # torch.float32 here
# -----------------------------------------------------------------------------
# UNet + scheduler (R-Flow). IMPORTANT: in_channels = latent_channels * 3
# -----------------------------------------------------------------------------
print("\n>> Loading UNet …")
cfg = json.load(open(args.config_unet))
latent_channels = cfg.get("latent_channels", 4)
cfg["diffusion_def"]["in_channels"] = latent_channels * 3  # [noisy, t1n, t2f]
unet = instantiate(cfg["diffusion_def"], cfg).to(DEV).half()
unet_ckpt = torch.load(args.ckpt_unet, map_location="cpu", weights_only=False)
unet.load_state_dict(unet_ckpt.get("unet", unet_ckpt), strict=True)
unet.eval(); print("✅  UNet checkpoint loaded")

scheduler = RFlowScheduler(
    num_train_timesteps=1000,
    use_discrete_timesteps=True,
    sample_method="logit-normal",
    use_timestep_transform=True,
    base_img_size_numel=64 * 64 * 48,
    spatial_dim=3,
)

# -----------------------------------------------------------------------------
# Dataset – pairs latents (T1n, T2F) with T1c target
# -----------------------------------------------------------------------------
class LatentTripletDataset(Dataset):
    """
    Returns dict:
      target=z_t1c, cond=z_t1n, cond2=z_t2f, cid, split
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
        z_c1.zero_()

        return {
            "target": z_tgt, "cond": z_c1, "cond2": z_c2,
            "cid": mu_tgt.parent.name, "split": mu_tgt.parent.parent.name
        }

root = Path(args.latent_dir)
loader = DataLoader(
    LatentTripletDataset(root),
    batch_size=max(1, args.batch),
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------
psnr_metric = PSNRMetric(max_val=1.0)
ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)

def minmax01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # normalize per-sample (B dimension)
    dims = tuple(range(1, x.ndim))
    mn = x.amin(dim=dims, keepdim=True)
    mx = x.amax(dim=dims, keepdim=True)
    return (x - mn) / (mx - mn + eps)

def nmse(pred: torch.Tensor, gt: torch.Tensor, eps: float=1e-8) -> float:
    num = torch.sum((pred - gt) ** 2, dtype=torch.float32)
    den = torch.sum((gt ** 2), dtype=torch.float32)
    return (num / (den + eps)).item()

def psnr_db(pred: torch.Tensor, gt: torch.Tensor, eps: float=1e-12) -> float:
    mse = torch.mean((pred - gt) ** 2, dtype=torch.float32)
    if mse <= 0:
        return float("inf")
    return (20.0 * torch.log10(torch.tensor(1.0)) - 10.0 * torch.log10(mse + eps)).item()

def ncc(pred: torch.Tensor, gt: torch.Tensor, eps: float=1e-8) -> float:
    p = pred.float(); g = gt.float()
    pm = p.mean(); gm = g.mean()
    pv = p - pm;   gv = g - gm
    denom = torch.sqrt((pv * pv).sum()) * torch.sqrt((gv * gv).sum())
    if denom < eps:
        return float("nan")
    return (pv * gv).sum().div(denom + eps).item()

def ssim3d(vol_pred: torch.Tensor, vol_gt: torch.Tensor) -> float:
    """
    vol_*: 3D tensors shaped (D,H,W) already normalized to [0,1]
    returns scalar SSIM
    """
    with torch.no_grad():
        s = ssim_metric(vol_pred.unsqueeze(0).unsqueeze(0),
                        vol_gt.unsqueeze(0).unsqueeze(0))
        return float(s.mean().item())

def apply_flips(vol: torch.Tensor, flip_x: bool=False, flip_y: bool=False, flip_z: bool=False) -> torch.Tensor:
    """
    Flip a 5D tensor shaped (B, C, D, H, W).
    x ≙ W (last dim), y ≙ H (second-to-last), z ≙ D (third-to-last).
    """
    dims = []
    if flip_z: dims.append(-3)
    if flip_y: dims.append(-2)
    if flip_x: dims.append(-1)
    return torch.flip(vol, dims=dims) if dims else vol

def load_t1c_affine(brats_root: Path, split: str, cid: str) -> np.ndarray:
    """
    Locate the T1c NIfTI for this case and return its affine.
    """
    cand = brats_root / split / cid / f"{cid}-t1c.nii.gz"
    if not cand.exists():
        # fallback: first *-t1c.nii.gz within the case folder
        case_dir = (brats_root / split / cid)
        matches = sorted(case_dir.glob("*-t1c.nii.gz"))
        if not matches:
            raise FileNotFoundError(f"No T1c NIfTI found for {split}/{cid} under {case_dir}")
        cand = matches[0]
    img = nib.load(str(cand))
    return img.affine

# -----------------------------------------------------------------------------
# CSV header
# -----------------------------------------------------------------------------
csv_path = OUT / "metrics_rflow_t1n_t2f.csv"
write_header = not csv_path.exists()
csv_f = open(csv_path, "a", newline="")
csv_w = csv.writer(csv_f)
if write_header:
    csv_w.writerow([
        "cid","split",
        "NMSE","PSNR_dB","NCC","SSIM",
        "pred_nii_path","pred_latent_path"
    ])

# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
print("\n>> Inference …")
brats_root = Path(args.brats_root)

with torch.no_grad():
    for batch in tqdm(loader, unit="vol"):
        z_cond  = batch["cond"].to(DEV)   # T1n latent
        z_cond2 = batch["cond2"].to(DEV)  # T2F latent
        z_tgt   = batch["target"].to(DEV) # T1c latent (for decoding GT)

        B = z_cond.size(0)
        z_curr = torch.randn_like(z_cond)  # start from noise

        # set scheduler timesteps for this spatial size
        num_vox = int(np.prod(z_curr.shape[-3:]))
        scheduler.set_timesteps(num_inference_steps=args.steps, input_img_size_numel=num_vox)
        t = scheduler.timesteps.to(DEV)
        t_next = torch.cat((t[1:], t.new_tensor([0])))

        # denoising loop
        for ts, tsn in tqdm(zip(t, t_next), total=len(t), desc="Denoising", leave=False):
            # scalar timesteps for the scheduler, batched timesteps for the UNet conditioning
            ts_scalar  = float(ts.item())
            tsn_scalar = float(tsn.item())
            ts_b  = ts.expand(B)
            unet_in = torch.cat([z_curr, z_cond, z_cond2], dim=1)  # matches training
            vel = unet(x=unet_in, timesteps=ts_b)
            z_curr, _ = scheduler.step(vel, ts_scalar, z_curr, tsn_scalar)


        z_pred = z_pred.to(device=ae_dev, dtype=ae_dt)
        z_gt = z_gt.to(device=ae_dev, dtype=ae_dt)
        # Decode to image space
        img_pred = autoencoder.decode(z_pred).float().cpu()  # B,1,D,H,W
        img_gt   = autoencoder.decode(z_tgt).float().cpu()

        img_pred = apply_flips(img_pred, args.flip_x, args.flip_y, args.flip_z)
        img_gt = apply_flips(img_gt, args.flip_x, args.flip_y, args.flip_z)

        # Normalize to [0,1] per volume before metrics
        img_pred_n = minmax01(img_pred)
        img_gt_n   = minmax01(img_gt)

        # Per-case bookkeeping
        for i in range(B):
            cid   = batch["cid"][i]
            split = batch["split"][i]
            case_dir = OUT / split / cid
            case_dir.mkdir(parents=True, exist_ok=True)

            # Whole-volume metrics
            vol_pred = img_pred_n[i, 0]  # D,H,W
            vol_gt   = img_gt_n[i, 0]

            nmse_whole = nmse(vol_pred, vol_gt)
            psnr_whole = psnr_db(vol_pred, vol_gt)
            ncc_whole  = ncc(vol_pred, vol_gt)
            ssim_whole = ssim3d(vol_pred, vol_gt)

            print(
                f"[{split}/{cid}] "
                f"NMSE={nmse_whole:.6f}, PSNR={psnr_whole:.2f} dB, NCC={ncc_whole:.4f}, SSIM={ssim_whole:.4f}"
            )

            # Save outputs
            lat_path = ""
            if args.save_latents:
                lat_path = str(case_dir / f"{cid}_t1c_pred_latent.pt")
                torch.save(z_pred[i].cpu(), lat_path)

            nii_path = ""
            if args.save_nifti:
                # write prediction with **T1c affine**
                nii_path = str(case_dir / f"{cid}_t1c_pred.nii.gz")
                arr = img_pred[i].cpu().numpy()  # 1,D,H,W
                affine = load_t1c_affine(brats_root, split, cid)
                writer = NibabelWriter()
                writer.set_data_array(arr, channel_dim=0)
                writer.set_metadata({"affine": affine, "original_affine": affine})
                writer.write(nii_path, verbose=False)

            # CSV row
            csv_w.writerow([
                cid, split,
                f"{nmse_whole:.6f}", f"{psnr_whole:.3f}", f"{ncc_whole:.5f}", f"{ssim_whole:.5f}",
                nii_path, lat_path
            ])
            csv_f.flush()

csv_f.close()
print("\n✅ Done. Metrics saved to:", csv_path)
print("   Output dir:", OUT.resolve())
