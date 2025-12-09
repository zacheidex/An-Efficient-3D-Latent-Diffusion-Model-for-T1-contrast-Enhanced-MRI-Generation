#!/usr/bin/env python3
"""
train_gan_t1n_t2f_seg.py – pix2pix‑style conditional GAN in latent space
=======================================================================
* Keeps your **same DiffusionModelUNet** backbone as the **generator** (no diffusion)
* Adds a lightweight 3D **PatchGAN discriminator**
* Conditions on [t1n, t2f, seg] latents and predicts **t1c latent**
* Logs G/D losses to CSV and plots a loss curve; saves latest & best checkpoints

Usage (example):
  python train_gan_t1n_t2f_seg.py -config ./config/config_train_16g.json \
         --latent_dir ./z_latent_maps --output_dir ./gan_t1n_t2f_seg

This script is a GAN refactor of your diffusion trainer; dataset paths and
file naming conventions are unchanged.
"""
from __future__ import annotations
import argparse, csv, importlib, json, math, os, random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# ----------------------------- config helpers -----------------------------

def _resolve(cfg: Dict[str, Any], root: Dict[str, Any]):
    if isinstance(cfg, dict):
        return {k: _resolve(v, root) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_resolve(v, root) for v in cfg]
    if isinstance(cfg, str):
        if cfg.startswith("@"):  # literal reference
            return root[cfg[1:]]
        if cfg.startswith("$@"):
            return root[cfg[2:]]
    return cfg


def instantiate(component_cfg: Dict[str, Any], root_cfg: Dict[str, Any]):
    comp = component_cfg.copy()
    target = comp.pop("_target_")
    kwargs = _resolve(comp, root_cfg)
    module_path, cls_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(**kwargs)

# ------------------------------- dataset ----------------------------------

class LatentPairDataset(Dataset):
    def __init__(self, split_dir: Path):
        self._items: List[tuple[Path, Path, Path, Path, Path, Path, Path, Path]] = []
        for mu_tgt in split_dir.rglob("*-t1c_z_mu.pt"):
            sig_tgt = mu_tgt.with_name(mu_tgt.name.replace("_z_mu.pt", "_z_sigma.pt"))

            mu_cond = mu_tgt.with_name(mu_tgt.name.replace("-t1c_z_mu.pt", "-t1n_z_mu.pt"))
            sig_cond = mu_cond.with_name(mu_cond.name.replace("_z_mu.pt", "_z_sigma.pt"))

            mu_cond_2 = mu_tgt.with_name(mu_tgt.name.replace("-t1c_z_mu.pt", "-t2f_z_mu.pt"))
            sig_cond_2 = mu_cond_2.with_name(mu_cond_2.name.replace("_z_mu.pt", "_z_sigma.pt"))

            if all(p.exists() for p in (mu_tgt, sig_tgt, mu_cond, sig_cond, mu_cond_2, sig_cond_2)):
                self._items.append((mu_tgt, sig_tgt, mu_cond, sig_cond, mu_cond_2, sig_cond_2))
        if not self._items:
            raise RuntimeError(f"No latent files found under {split_dir}")

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        (mu_tgt, sig_tgt,
         mu_c1,  sig_c1,
         mu_c2,  sig_c2) = self._items[idx]

        μ_t,  σ_t,  μ_c1, σ_c1, μ_c2, σ_c2 = [torch.load(p) for p in (
            mu_tgt, sig_tgt, mu_c1, sig_c1, mu_c2, sig_c2
        )]

        z_tgt   = μ_t  + σ_t  * torch.randn_like(μ_t)
        z_c1    = μ_c1 + σ_c1 * torch.randn_like(μ_c1)
        z_c2    = μ_c2 + σ_c2 * torch.randn_like(μ_c2)

        cid = mu_tgt.parent.name
        split = mu_tgt.parent.parent.name
        return {"target": z_tgt, "cond1": z_c1, "cond2": z_c2, "cid": cid, "split": split}

# ------------------------------- modules ----------------------------------

class GeneratorUNetWrapper(nn.Module):
    """Wrap MONAI DiffusionModelUNet to ignore timesteps (feed zeros)."""
    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet
    def forward(self, x):
        # DiffusionModelUNet signature: (x, timesteps)
        t = torch.zeros((x.shape[0],), dtype=torch.long, device=x.device)
        return self.unet(x, t)

class PatchDiscriminator3D(nn.Module):
    """3D PatchGAN (70×70×70-ish receptive field) for latent tensors.
    Input channels = channels(input conditions) + channels(target or fake).
    """
    def __init__(self, in_channels: int, ndf: int = 64, num_layers: int = 4):
        super().__init__()
        layers = []
        # layer 0 (no norm)
        layers += [nn.Conv3d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
                   nn.LeakyReLU(0.2, inplace=True)]
        nf_mult = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4,
                          stride=2 if n < num_layers - 1 else 1, padding=1, bias=False),
                nn.InstanceNorm3d(ndf * nf_mult, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        # last conv → 1‑chan patch logits
        layers += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)  # (B,1,D',H',W') logits

# ------------------------------- training ---------------------------------

@dataclass
class GANHyper:
    lr_G: float = 1e-4
    lr_D: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_L1: float = 100.0
    max_epochs: int = 101
    val_interval: int = 2
    batch_size: int = 4


def seed_all(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser(description="Latent pix2pix trainer (3D)")
    p.add_argument("-env", default="./config/environment.json")
    p.add_argument("-config", default="./config/config_train_16g.json")
    p.add_argument("-g", "--gpus", type=int, default=1)
    p.add_argument("--latent_dir", default="./z_latent_maps")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="./gan_t1n_t2f")
    args = p.parse_args()

    with open(args.config) as f: cfg = json.load(f)
    # slap on sensible GAN defaults if missing
    gt = cfg.get("gan_train", {})
    hyper = GANHyper(
        lr_G=gt.get("lr_G", 1e-4), lr_D=gt.get("lr_D", 1e-4),
        beta1=gt.get("beta1", 0.5), beta2=gt.get("beta2", 0.999),
        lambda_L1=gt.get("lambda_L1", 100.0),
        max_epochs=gt.get("max_epochs", cfg.get("diffusion_train", {}).get("max_epochs", 101)),
        val_interval=gt.get("val_interval", cfg.get("diffusion_train", {}).get("val_interval", 2)),
        batch_size=gt.get("batch_size", cfg.get("diffusion_train", {}).get("batch_size", 4)),
    )

    seed_all(); device = torch.device(args.device)

    # ---------------- models ----------------
    latent_ch = int(cfg["latent_channels"])  # e.g., 4
    cond_ch_total = latent_ch * 2            # t1n + t2f + seg

    # Reuse your UNet config as the GENERATOR backbone, just fix channels
    gen_cfg = cfg["diffusion_def"].copy()
    gen_cfg["in_channels"]  = cond_ch_total
    gen_cfg["out_channels"] = latent_ch

    print(">> Instantiating Generator (DiffusionModelUNet backbone)…")
    gen_backbone = instantiate(gen_cfg, cfg).to(device)
    G = GeneratorUNetWrapper(gen_backbone).to(device)

    # PatchGAN discriminator sees [conditions || target_or_fake]
    D_in_ch = cond_ch_total + latent_ch
    D = PatchDiscriminator3D(in_channels=D_in_ch, ndf=64, num_layers=4).to(device)

    # ---------------- opt & loss ----------------
    optG = AdamW(G.parameters(), lr=hyper.lr_G, betas=(hyper.beta1, hyper.beta2), weight_decay=1e-4)
    optD = AdamW(D.parameters(), lr=hyper.lr_D, betas=(hyper.beta1, hyper.beta2), weight_decay=1e-4)
    scalerG, scalerD = GradScaler(), GradScaler()
    bce = nn.BCEWithLogitsLoss()

    # ---------------- data ----------------
    root = Path(args.latent_dir)
    train_loader = DataLoader(LatentPairDataset(root / "train"), batch_size=hyper.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(LatentPairDataset(root / "val"),   batch_size=hyper.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ---------------- logging ----------------
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    csv_path, png_path = out / "loss_history.csv", out / "loss_curve.png"
    fieldnames = ["epoch", "step", "g_total", "g_adv", "g_l1", "d_real", "d_fake", "val_l1"]
    with csv_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    history: List[Dict[str, Any]] = []
    best_val = math.inf
    step = 0

    # ---------------- training loop ----------------
    for epoch in range(hyper.max_epochs):
        G.train(); D.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            tgt  = batch["target"].to(device)
            c1   = batch["cond1"].to(device)
            c2   = batch["cond2"].to(device)
            cond = torch.cat([c1, c2], dim=1)

            # ------------------- update D -------------------
            optD.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.float16):
                fake = G(cond)  # (B, latent_ch, D,H,W)
                # real pair
                real_in = torch.cat([cond, tgt], dim=1)
                fake_in = torch.cat([cond, fake.detach()], dim=1)
                pred_real = D(real_in)
                pred_fake = D(fake_in)
                valid = torch.ones_like(pred_real)
                fake_lbl = torch.zeros_like(pred_fake)
                d_real = bce(pred_real, valid)
                d_fake = bce(pred_fake, fake_lbl)
                d_loss = 0.5 * (d_real + d_fake)
            scalerD.scale(d_loss).backward()
            scalerD.step(optD)
            scalerD.update()

            # ------------------- update G -------------------
            optG.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.float16):
                fake = G(cond)
                fake_in = torch.cat([cond, fake], dim=1)
                pred_fake = D(fake_in)
                g_adv = bce(pred_fake, torch.ones_like(pred_fake))
                g_l1  = F.l1_loss(fake, tgt)
                g_total = g_adv + hyper.lambda_L1 * g_l1
            scalerG.scale(g_total).backward()
            scalerG.step(optG)
            scalerG.update()

            pbar.set_postfix(g=f"{g_total.item():.3f}", d=f"{d_loss.item():.3f}")
            history.append({"epoch": epoch, "step": step, "g_total": g_total.item(),
                            "g_adv": g_adv.item(), "g_l1": g_l1.item(),
                            "d_real": d_real.item(), "d_fake": d_fake.item(), "val_l1": None})
            with csv_path.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(history[-1])
            step += 1

        # ---------------- validation ----------------
        if (epoch + 1) % hyper.val_interval == 0:
            G.eval()
            vals: List[float] = []
            with torch.no_grad(), autocast(device_type=device.type, dtype=torch.float16):
                for batch in tqdm(val_loader, desc=f"Val {epoch}", leave=False):
                    tgt  = batch["target"].to(device)
                    c1   = batch["cond1"].to(device)
                    c2   = batch["cond2"].to(device)
                    cond = torch.cat([c1, c2], dim=1)
                    pred = G(cond)
                    vals.append(F.l1_loss(pred, tgt).item())
            val_l1 = float(np.mean(vals)) if vals else float("inf")
            print(f"\n>> Validation | epoch {epoch} | L1 {val_l1:.4f}")

            # save latest
            torch.save({"G": G.state_dict(), "epoch": epoch, "val_l1": val_l1}, out / "gen_latest.pt")
            torch.save({"D": D.state_dict(), "epoch": epoch, "val_l1": val_l1}, out / "dis_latest.pt")
            if val_l1 < best_val:
                best_val = val_l1
                torch.save({"G": G.state_dict(), "epoch": epoch, "val_l1": val_l1}, out / "gen_best.pt")
                print(f"   ↓ new best generator saved (L1 {val_l1:.4f})")

            # log a val row (no train metrics)
            history.append({"epoch": epoch, "step": step, "g_total": None, "g_adv": None, "g_l1": None,
                            "d_real": None, "d_fake": None, "val_l1": val_l1})
            with csv_path.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(history[-1])

            # plot
            tr_steps  = [h["step"] for h in history if h["g_total"] is not None]
            tr_g      = [h["g_total"] for h in history if h["g_total"] is not None]
            tr_d      = [ (h["d_real"] + h["d_fake"]) * 0.5 for h in history if h["d_real"] is not None]
            val_steps = [h["step"] for h in history if h["val_l1"] is not None]
            val_vals  = [h["val_l1"] for h in history if h["val_l1"] is not None]

            plt.figure(figsize=(7,4))
            if tr_steps:
                plt.plot(tr_steps, tr_g, label="G total")
                plt.plot(tr_steps, tr_d, label="D avg")
            if val_steps:
                plt.plot(val_steps, val_vals, label="val L1")
            plt.xlabel("step"); plt.ylabel("loss")
            plt.legend(); plt.tight_layout()
            plt.title(f"GAN training ({datetime.now().strftime('%m-%d %H:%M')})")
            plt.savefig(png_path); plt.close()

    print("Training finished! Checkpoints are in:", out)


if __name__ == "__main__":
    main()
