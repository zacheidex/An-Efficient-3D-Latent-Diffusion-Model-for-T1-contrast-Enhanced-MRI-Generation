#!/usr/bin/env python3
"""
R-Flow (DiT) latent→latent inference (T1n + T2F → T1c), **no segmentation**.
- Whole‑volume metrics: NMSE, PSNR(dB), NCC, SSIM
- Writes prediction NIfTI using the affine from the **T1c** NIfTI in Brats_preprocessed
- Input ordering to the DiT matches training: [x_t, T1n, T2F]  (in_channels = latent_channels * 3)
- **No patching**: runs on the whole latent volume in one shot (to address your “patches aren’t working” issue).

Usage (example):
  python test_rflow_t1_flair_dit_noseg.py \
    --ckpt-ae autoencoder_epoch273.pt \
    --ckpt-unet ./rflow_DiT_t1_flair/unet_latest.pt \
    --config-unet ./config/config_train_vit.json \
    --latent-dir ./z_latent_maps/test \
    --brats-root ./Brats_preprocessed \
    --out-dir ./rflow_dit_t1n_t2f_eval_noseg \
    --device cuda:0 --steps 200
"""

import argparse, json, csv, importlib
from pathlib import Path
from typing import Any, Dict
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast

import nibabel as nib
from monai.config import print_config
from monai.utils import set_determinism
from monai.data import NibabelWriter
from monai.metrics import SSIMMetric
from monai.networks.schedulers import RFlowScheduler

# Your DiT wrapper (expects forward(x, t))
from dit3d_wrapper import DiT3DWrapper

# -----------------------------------------------------------------------------
# Mini “instantiate” helper (same style as your training code)
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
ap.add_argument("--ckpt-ae",      default="autoencoder_epoch273.pt")
ap.add_argument("--ckpt-unet",    default="./rflow_DiT_t1_flair/unet_latest.pt", help="DiT R-Flow checkpoint")
ap.add_argument("--config-unet",  default="./config/config_train_vit.json")
ap.add_argument("--latent-dir",   default="./z_latent_maps/test")
ap.add_argument("--brats-root",   default="/data/zeidex/BraTS_preprocessed")
ap.add_argument("--out-dir",      default="./rflow_dit_t1n_t2f_eval_noseg")
ap.add_argument("--device",       default="cuda:0")
ap.add_argument("--batch",        type=int, default=1)
ap.add_argument("--steps",        type=int, default=200, help="R-Flow inference steps")
ap.add_argument("--save-nifti",   default=True)
ap.add_argument("--save-latents", action="store_true")
ap.add_argument("--seed",         type=int, default=0)
# Optional axis flips (if needed for orientation corrections)
ap.add_argument("--flip-x", action="store_true")  # width
ap.add_argument("--flip-y", action="store_true")  # height
ap.add_argument("--flip-z", default=True)          # depth
args = ap.parse_args()

set_determinism(args.seed)
DEV = torch.device(args.device)
OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# AutoEncoder‑KL (keep fp32 at inference for stability)
# -----------------------------------------------------------------------------
print("\n>> Loading AutoEncoder‑KL …")
ENV_JSON   = "./maisi/configs/environment_maisi_vae_train.json"
ARCH_JSON  = "./maisi/configs/config_maisi3d-rflow.json"
# Mirror how you construct it elsewhere
env_args = argparse.Namespace(**json.load(open(ENV_JSON)))
arch_dict = json.load(open(ARCH_JSON))
for k, v in arch_dict.items():
    setattr(env_args, k, v)
# force num_splits=1 for inference
env_args.autoencoder_def["num_splits"] = 1
from maisi.scripts.utils import define_instance  # noqa: E402
autoencoder = define_instance(env_args, "autoencoder_def").to(DEV).half()
# load ckpt (supports raw or keyed)
ae_ckpt = torch.load(args.ckpt_ae, map_location="cpu")
if isinstance(ae_ckpt, dict) and "autoencoder" in ae_ckpt:
    autoencoder.load_state_dict(ae_ckpt["autoencoder"], strict=True)
else:
    autoencoder.load_state_dict(ae_ckpt, strict=True)
autoencoder.eval(); print("✅  AutoEncoder ready (fp32)")

# -----------------------------------------------------------------------------
# DiT model (configured like the trainer); run in fp32 to avoid AMP gotchas
# -----------------------------------------------------------------------------
print("\n>> Loading DiT (R‑Flow) …")
cfg = json.load(open(args.config_unet))
latent_ch = int(cfg.get("latent_channels", 4))
in_ch  = latent_ch * 3    # [x_t, T1n, T2F]
out_ch = latent_ch
unet = DiT3DWrapper(
    in_channels=in_ch,
    out_channels=out_ch,
    input_size=(64, 64, 48),
    patch_size=4,
    hidden_size=768,
    depth=16,
    num_heads=8,
    window_size=4,
    window_block_indexes=[0, 3, 6, 9]
)
unet_ckpt = torch.load(args.ckpt_unet, map_location="cpu")
key = "unet" if isinstance(unet_ckpt, dict) and "unet" in unet_ckpt else None
unet.load_state_dict(unet_ckpt[key] if key else unet_ckpt, strict=True)
unet.to(DEV).eval(); print("✅  DiT checkpoint loaded (fp32)")

scheduler = RFlowScheduler(
    num_train_timesteps=1000,
    use_discrete_timesteps=True,
    sample_method="logit-normal",
    use_timestep_transform=True,
    base_img_size_numel=64 * 64 * 48,
    spatial_dim=3,
)

# -----------------------------------------------------------------------------
# Dataset (T1n, T2F) → T1c latents; operates on full volumes
# -----------------------------------------------------------------------------
class LatentTripletDataset(Dataset):
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
        μt, σt, μ1, σ1, μ2, σ2 = [torch.load(p) for p in (mu_tgt, sig_tgt, mu_c1, sig_c1, mu_c2, sig_c2)]
        z_tgt  = μt + σt * torch.randn_like(μt)
        z_c1   = μ1 + σ1 * torch.randn_like(μ1)
        z_c2   = μ2 + σ2 * torch.randn_like(μ2)
        return {"target": z_tgt, "cond": z_c1, "cond2": z_c2,
                "cid": mu_tgt.parent.name, "split": mu_tgt.parent.parent.name}

root = Path(args.latent_dir)
loader = DataLoader(
    LatentTripletDataset(root),
    batch_size=max(1, args.batch),
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# -----------------------------------------------------------------------------
# Metrics & helpers
# -----------------------------------------------------------------------------
ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)

def minmax01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
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
    with torch.no_grad():
        s = ssim_metric(vol_pred.unsqueeze(0).unsqueeze(0),
                        vol_gt.unsqueeze(0).unsqueeze(0))
        return float(s.mean().item())

def apply_flips(vol: torch.Tensor, flip_x=False, flip_y=False, flip_z=False) -> torch.Tensor:
    dims = []
    if flip_z: dims.append(-3)
    if flip_y: dims.append(-2)
    if flip_x: dims.append(-1)
    return torch.flip(vol, dims=dims) if dims else vol

def load_t1c_affine(brats_root: Path, split: str, cid: str):
    cand = brats_root / split / cid / f"{cid}-t1c.nii.gz"
    if not cand.exists():
        case_dir = brats_root / split / cid
        matches = sorted(case_dir.glob("*-t1c.nii.gz"))
        if not matches:
            raise FileNotFoundError(f"No T1c NIfTI found for {split}/{cid} under {case_dir}")
        cand = matches[0]
    return nib.load(str(cand)).affine

def align_to_module(x, module):
    p = next(module.parameters())
    return x.to(device=p.device, dtype=p.dtype)
# -----------------------------------------------------------------------------
# CSV
# -----------------------------------------------------------------------------
csv_path = OUT / "metrics_rflow_dit_t1n_t2f.csv"
write_header = not csv_path.exists()
csv_f = open(csv_path, "a", newline="")
csv_w = csv.writer(csv_f)
if write_header:
    csv_w.writerow(["cid","split","NMSE","PSNR_dB","NCC","SSIM","pred_nii_path","pred_latent_path"])

# -----------------------------------------------------------------------------
# Inference (whole‑volume, dtype/device‑aligned)
# -----------------------------------------------------------------------------
print("\n>> Inference (R‑Flow, DiT, no‑seg) …")
brats_root = Path(args.brats_root)

with torch.no_grad():
    for batch in loader:
        z_cond  = batch["cond"].to(DEV)   # T1n latent
        z_cond2 = batch["cond2"].to(DEV)  # T2F latent
        z_tgt   = batch["target"].to(DEV) # T1c latent (for decoding GT)

        B = z_cond.size(0)
        z_curr = torch.randn_like(z_cond)  # start from noise

        # Configure scheduler for this spatial size
        num_vox = int(np.prod(z_curr.shape[-3:]))
        try:
            scheduler.set_timesteps(num_inference_steps=args.steps, input_img_size_numel=num_vox)
        except TypeError:
            scheduler.set_timesteps(args.steps, num_vox)

        t = scheduler.timesteps.float().to(DEV)
        t_next = torch.cat((t[1:], t.new_tensor([0])))

        # DiT forward expects (x, t). Keep everything fp32 and aligned to model params.
        model_dtype = next(unet.parameters()).dtype
        model_dev   = next(unet.parameters()).device

        for ts, tsn in zip(t, t_next):
            ts_b  = ts.expand(B)
            tsn_b = tsn.expand(B)
            unet_in = torch.cat([z_curr, z_cond, z_cond2], dim=1).to(device=model_dev, dtype=model_dtype)
            vel = unet(unet_in, ts_b)
            z_curr, _ = scheduler.step(vel, ts_b, z_curr, tsn_b)

        z_pred = z_curr



        # Decode (keep AE in fp32; disable autocast to avoid silent recasts)
        ae_param = next(autoencoder.parameters())
        ae_dev, ae_dt = ae_param.device, ae_param.dtype

        with autocast(device_type=DEV.type, enabled=False):  # avoid AMP re-casting to fp16
            z_aligned = align_to_module(z_pred, autoencoder)
            img_pred  = autoencoder.decode(z_aligned).float().cpu()  # B,1,D,H,W
            img_gt   = autoencoder.decode(z_tgt .to(device=ae_dev, dtype=ae_dt)).float().cpu()

        # Optional flips
        img_pred = apply_flips(img_pred, args.flip_x, args.flip_y, args.flip_z)
        img_gt   = apply_flips(img_gt,   args.flip_x, args.flip_y, args.flip_z)

        # Normalize to [0,1]
        img_pred_n = minmax01(img_pred)
        img_gt_n   = minmax01(img_gt)

        # Per‑case
        for i in range(img_pred_n.size(0)):
            cid, split = batch["cid"][i], batch["split"][i]
            case_dir = OUT / split / cid; case_dir.mkdir(parents=True, exist_ok=True)

            vol_p, vol_g = img_pred_n[i,0], img_gt_n[i,0]
            nmse_whole = nmse(vol_p, vol_g)
            psnr_whole = psnr_db(vol_p, vol_g)
            ncc_whole  = ncc(vol_p, vol_g)
            ssim_whole = ssim3d(vol_p, vol_g)

            print(f"[{split}/{cid}] NMSE={nmse_whole:.6f}, PSNR={psnr_whole:.2f} dB, NCC={ncc_whole:.4f}, SSIM={ssim_whole:.4f}")

            lat_path = ""
            if args.save_latents:
                lat_path = str(case_dir / f"{cid}_t1c_pred_latent.pt")
                torch.save(z_pred[i].cpu(), lat_path)

            nii_path = ""
            if args.save_nifti:
                nii_path = str(case_dir / f"{cid}_t1c_pred.nii.gz")
                arr = img_pred[i].cpu().numpy()  # 1,D,H,W
                affine = load_t1c_affine(Path(args.brats_root), split, cid)
                writer = NibabelWriter()
                writer.set_data_array(arr, channel_dim=0)
                writer.set_metadata({"affine": affine, "original_affine": affine})
                writer.write(nii_path, verbose=False)

            csv_w.writerow([cid, split, f"{nmse_whole:.6f}", f"{psnr_whole:.3f}", f"{ncc_whole:.5f}", f"{ssim_whole:.5f}", nii_path, lat_path])
            csv_f.flush()

csv_f.close()
print("\n✅ Done. Metrics saved to:", csv_path)
print("   Output dir:", OUT.resolve())
