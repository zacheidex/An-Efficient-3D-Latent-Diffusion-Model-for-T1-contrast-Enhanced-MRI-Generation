#!/usr/bin/env python3
"""
pix2pix (GAN) latent→latent **inference** (T1n + T2F → T1c) — NO segmentation inputs/metrics.

What it does
------------
- Loads your generator (DiffusionModelUNet backbone) and runs a single forward pass in latent space.
- Whole-volume metrics only: NMSE, PSNR(dB), NCC, SSIM.
- Writes prediction NIfTI using the affine from the **T1c** NIfTI in Brats_preprocessed.
- If your generator checkpoint expects 3 conditions (T1n, T2F, Seg) it will **auto‑fallback**
  by adding a zero segmentation latent so you don't need seg files.

Expected latents
----------------
- <cid>-t1n_z_mu.pt, <cid>-t1n_z_sigma.pt
- <cid>-t2f_z_mu.pt, <cid>-t2f_z_sigma.pt
- <cid>-t1c_z_mu.pt, <cid>-t1c_z_sigma.pt    (for decoding GT & metrics)

"""
import argparse, json, importlib, csv
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast

import nibabel as nib
from monai.config import print_config
from monai.utils import set_determinism
from monai.data import NibabelWriter
from monai.metrics import SSIMMetric

def _resolve(cfg: Dict[str, Any], root: Dict[str, Any]):
    if isinstance(cfg, dict):
        return {k: _resolve(v, root) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_resolve(v, root) for v in cfg]
    if isinstance(cfg, str):
        if cfg.startswith("@"): return root[cfg[1:]]
        if cfg.startswith("$@"): return root[cfg[2:]]
    return cfg

def instantiate(cfg: Dict[str, Any], root: Dict[str, Any]):
    comp = cfg.copy()
    target = comp.pop("_target_")
    kwargs = _resolve(comp, root)
    module, cls = target.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)(**kwargs)

class GeneratorUNetWrapper(nn.Module):
    def __init__(self, unet: nn.Module):
        super().__init__(); self.unet = unet
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.zeros((x.shape[0],), dtype=torch.long, device=x.device)
        return self.unet(x, t)

class LatentPairDataset(Dataset):
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
        mu_tgt, sig_tgt, mu_c1, sig_c1, mu_c2, sig_c2 = self._items[idx]
        μt, σt, μ1, σ1, μ2, σ2 = [torch.load(p) for p in (mu_tgt, sig_tgt, mu_c1, sig_c1, mu_c2, sig_c2)]
        z_tgt = μt + σt * torch.randn_like(μt)
        z_c1  = μ1 + σ1 * torch.randn_like(μ1)
        z_c2  = μ2 + σ2 * torch.randn_like(μ2)
        return {"target": z_tgt, "cond": z_c1, "cond2": z_c2, "cid": mu_tgt.parent.name, "split": mu_tgt.parent.parent.name}

ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)
def minmax01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dims = tuple(range(1, x.ndim)); mn = x.amin(dim=dims, keepdim=True); mx = x.amax(dim=dims, keepdim=True)
    return (x - mn) / (mx - mn + eps)
def nmse(pred: torch.Tensor, gt: torch.Tensor, eps: float=1e-8) -> float:
    num = torch.sum((pred - gt) ** 2, dtype=torch.float32); den = torch.sum((gt ** 2), dtype=torch.float32)
    return (num / (den + eps)).item()
def psnr_db(pred: torch.Tensor, gt: torch.Tensor, eps: float=1e-12) -> float:
    mse = torch.mean((pred - gt) ** 2, dtype=torch.float32); 
    if mse <= 0: return float("inf")
    return (20.0 * torch.log10(torch.tensor(1.0)) - 10.0 * torch.log10(mse + eps)).item()
def ncc(pred: torch.Tensor, gt: torch.Tensor, eps: float=1e-8) -> float:
    p = pred.float(); g = gt.float(); pm = p.mean(); gm = g.mean(); pv = p - pm; gv = g - gm
    denom = torch.sqrt((pv * pv).sum()) * torch.sqrt((gv * gv).sum())
    if denom < eps: return float("nan")
    return (pv * gv).sum().div(denom + eps).item()
def ssim3d(vol_pred: torch.Tensor, vol_gt: torch.Tensor) -> float:
    with torch.no_grad():
        s = ssim_metric(vol_pred.unsqueeze(0).unsqueeze(0), vol_gt.unsqueeze(0).unsqueeze(0))
        return float(s.mean().item())
def apply_flips(vol: torch.Tensor, flip_x=False, flip_y=False, flip_z=False) -> torch.Tensor:
    dims = []; 
    if flip_z: dims.append(-3)
    if flip_y: dims.append(-2)
    if flip_x: dims.append(-1)
    return torch.flip(vol, dims=dims) if dims else vol
def load_t1c_affine(brats_root: Path, split: str, cid: str):
    cand = brats_root / split / cid / f"{cid}-t1c.nii.gz"
    if not cand.exists():
        case_dir = brats_root / split / cid
        matches = sorted(case_dir.glob("*-t1c.nii.gz"))
        if not matches: raise FileNotFoundError(f"No T1c NIfTI found for {split}/{cid} under {case_dir}")
        cand = matches[0]
    return nib.load(str(cand)).affine

def main():
    print_config()
    ap = argparse.ArgumentParser(description="pix2pix latent inference (no-seg)")
    ap.add_argument("--ckpt-ae",     default="autoencoder_epoch273.pt")
    ap.add_argument("--ckpt-gen",    default="./gan_t1n_t2f/gen_latest.pt")
    ap.add_argument("--config-unet", default="./config/config_train_16g.json")
    ap.add_argument("--latent-dir",  default="./z_latent_maps/test")
    ap.add_argument("--brats-root",  default="/data/zeidex/BraTS_preprocessed")
    ap.add_argument("--out-dir",     default="./pix2pix_t1n_t2f_gt_only")
    ap.add_argument("--device",      default="cuda:0")
    ap.add_argument("--batch",       type=int, default=1)
    ap.add_argument("--save-nifti",  default=True)
    ap.add_argument("--save-latents",action="store_true")
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--flip-x", action="store_true")
    ap.add_argument("--flip-y", action="store_true")
    ap.add_argument("--flip-z", default=True)
    args = ap.parse_args()

    set_determinism(args.seed)
    DEV = torch.device(args.device)
    OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)

    print("\n>> Loading AutoEncoder-KL …")
    ENV_JSON   = "./maisi/configs/environment_maisi_vae_train.json"
    ARCH_JSON  = "./maisi/configs/config_maisi3d-rflow.json"
    env_args = argparse.Namespace(**json.load(open(ENV_JSON)))
    arch_dict = json.load(open(ARCH_JSON))
    for k, v in arch_dict.items(): setattr(env_args, k, v)
    env_args.autoencoder_def["num_splits"] = 1
    from maisi.scripts.utils import define_instance
    autoencoder = define_instance(env_args, "autoencoder_def").to(DEV).half()
    ae_ckpt     = torch.load(args.ckpt_ae, map_location="cpu")
    autoencoder.load_state_dict(ae_ckpt.get("autoencoder", ae_ckpt), strict=True)
    autoencoder.eval(); print("✅  AutoEncoder ready (fp16)")

    print("\n>> Loading Generator (UNet backbone) …")
    cfg = json.load(open(args.config_unet))
    latent_ch = int(cfg.get("latent_channels", 4))

    gen_cfg = cfg["diffusion_def"].copy()
    gen_cfg["in_channels"]  = latent_ch * 2
    gen_cfg["out_channels"] = latent_ch

    def make_G(local_cfg):
        unet = instantiate(local_cfg, cfg).to(DEV).half()
        return GeneratorUNetWrapper(unet).to(DEV).half()

    G = make_G(gen_cfg)
    ckpt = torch.load(args.ckpt_gen, map_location="cpu")
    sd = ckpt["G"] if isinstance(ckpt, dict) and "G" in ckpt else ckpt

    need_zero_seg = False
    try:
        G.load_state_dict(sd, strict=True)
        print("✅  Loaded generator with 2-condition input (T1n,T2F).")
    except Exception:
        print("! 2-cond load failed; trying 3-cond (zero‑seg fallback).")
        gen_cfg3 = cfg["diffusion_def"].copy()
        gen_cfg3["in_channels"]  = latent_ch * 3
        gen_cfg3["out_channels"] = latent_ch
        G = make_G(gen_cfg3)
        G.load_state_dict(sd, strict=True)
        need_zero_seg = True
        print("✅  Loaded 3‑cond generator. Will feed zero seg channels.")

    root = Path(args.latent_dir)
    loader = DataLoader(LatentPairDataset(root), batch_size=max(1, args.batch),
                        shuffle=False, num_workers=2, pin_memory=True)

    csv_path = OUT / "metrics_pix2pix_t1n_t2f.csv"
    write_header = not csv_path.exists()
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if write_header:
        csv_w.writerow(["cid","split","NMSE","PSNR_dB","NCC","SSIM","pred_nii_path","pred_latent_path","zero_seg_fallback"])

    print("\n>> Inference (pix2pix, no‑seg) …")
    brats_root = Path(args.brats_root)
    G.eval(); torch.set_grad_enabled(False)

    for batch in loader:
        z_c1  = batch["cond"].to(DEV)
        z_c2  = batch["cond2"].to(DEV)
        z_tgt = batch["target"].to(DEV)

        if need_zero_seg:
            B, C, D, H, W = z_c1.shape
            z_seg0 = torch.zeros((B, C, D, H, W), device=DEV, dtype=z_c1.dtype)
            z_in = torch.cat([z_c1, z_c2, z_seg0], dim=1)
        else:
            z_in = torch.cat([z_c1, z_c2], dim=1)

        with autocast(device_type=DEV.type, dtype=torch.float16):
            z_pred = G(z_in)

        img_pred = autoencoder.decode(z_pred).float().cpu()
        img_gt   = autoencoder.decode(z_tgt).float().cpu()

        img_pred = apply_flips(img_pred, args.flip_x, args.flip_y, args.flip_z)
        img_gt   = apply_flips(img_gt,   args.flip_x, args.flip_y, args.flip_z)

        img_pred_n = minmax01(img_pred); img_gt_n = minmax01(img_gt)

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
                '''
                nii_path = str(case_dir / f"{cid}_t1c_pred.nii.gz")
                arr = img_pred[i].cpu().numpy()
                affine = load_t1c_affine(Path(args.brats_root), split, cid)
                writer = NibabelWriter()
                writer.set_data_array(arr, channel_dim=0)
                writer.set_metadata({"affine": affine, "original_affine": affine})
                writer.write(nii_path, verbose=False)
                '''
                nii_path = str(case_dir / f"{cid}_t1c_gt.nii.gz")
                arr = img_gt[i].cpu().numpy()
                affine = load_t1c_affine(Path(args.brats_root), split, cid)
                writer = NibabelWriter()
                writer.set_data_array(arr, channel_dim=0)
                writer.set_metadata({"affine": affine, "original_affine": affine})
                writer.write(nii_path, verbose=False)


            csv_w.writerow([cid, split, f"{nmse_whole:.6f}", f"{psnr_whole:.3f}", f"{ncc_whole:.5f}", f"{ssim_whole:.5f}", nii_path, lat_path, int(need_zero_seg)])
            csv_f.flush()

    csv_f.close()
    print("\n✅ Done. Metrics saved to:", csv_path)
    print("   Output dir:", OUT.resolve())

if __name__ == "__main__":
    main()
