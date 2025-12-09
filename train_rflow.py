#!/usr/bin/env python3
"""
latent_diffusion_train_full.py – latent‑to‑latent R‑Flow trainer
==============================================================
* Logs **train** loss every epoch and **val** loss every `val_interval` epochs
* Appends both to *loss_history.csv* and redraws a loss curve PNG
* Saves checkpoints: latest, best (lowest val‑loss), and epoch snapshots every 50 epochs
"""
# --------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------
import argparse, json, math, os, random, importlib, csv
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm.auto import tqdm
from monai.networks.schedulers import RFlowScheduler
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def _resolve(cfg: Dict[str, Any], root: Dict[str, Any]):
    """Recursively replace strings starting with @ or $@ with values in root."""
    if isinstance(cfg, dict):
        return {k: _resolve(v, root) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_resolve(v, root) for v in cfg]
    if isinstance(cfg, str):
        if cfg.startswith("@"):  # literal reference
            return root[cfg[1:]]
        if cfg.startswith("$@"):  # numeric value in root
            return root[cfg[2:]]
    return cfg


def instantiate(component_cfg: Dict[str, Any], root_cfg: Dict[str, Any]):
    """Instantiate a class described by a MONAI‑style mini‑config dict."""
    comp = component_cfg.copy()
    target = comp.pop("_target_")
    kwargs = _resolve(comp, root_cfg)
    module_path, cls_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(**kwargs)

# --------------------------------------------------------------------------------------
# Dataset producing latent pairs
# --------------------------------------------------------------------------------------

class LatentPairDataset(Dataset):
    def __init__(self, split_dir: Path):
        self._triples: list[tuple[Path, Path, Path, Path, Path, Path]] = []
        for mu_tgt in split_dir.rglob("*-t1c_z_mu.pt"):
            sig_tgt = mu_tgt.with_name(mu_tgt.name.replace("_z_mu.pt", "_z_sigma.pt"))

            mu_cond = mu_tgt.with_name(mu_tgt.name.replace("-t1c_z_mu.pt", "-t1n_z_mu.pt"))
            sig_cond = mu_cond.with_name(mu_cond.name.replace("_z_mu.pt", "_z_sigma.pt"))

            mu_seg  = mu_tgt.with_name(mu_tgt.name.replace("-t1c_z_mu.pt", "-t2f_z_mu.pt"))
            sig_seg = mu_seg.with_name(mu_seg.name.replace("_z_mu.pt", "_z_sigma.pt"))

            if all(p.exists() for p in (sig_tgt, mu_cond, sig_cond, mu_seg, sig_seg)):
                self._triples.append((mu_tgt, sig_tgt, mu_cond, sig_cond, mu_seg, sig_seg))
        if not self._triples:
            raise RuntimeError(f"No latent files found under {split_dir}")

    def __len__(self):
        return len(self._triples)

    def __getitem__(self, idx):
        mu_tgt, sig_tgt, mu_cond, sig_cond, mu_seg, sig_seg = self._triples[idx]
        μ_t, σ_t, μ_c, σ_c, μ_s, σ_s = [torch.load(p) for p in (mu_tgt, sig_tgt, mu_cond, sig_cond, mu_seg, sig_seg)]
        z_tgt = μ_t + σ_t * torch.randn_like(μ_t)
        z_cond = μ_c + σ_c * torch.randn_like(μ_c)
        z_seg  = μ_s + σ_s * torch.randn_like(μ_s)
        # Get identifiers
        cid = mu_tgt.parent.name   # Folder name (usually the case id)
        split = mu_tgt.parent.parent.name   # Parent folder (usually 'test', 'val', etc)
        return {"target": z_tgt, "cond": z_cond, "seg": z_seg, "cid": cid, "split": split}


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------------------------------------------------------------------------------
# Main training function
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Latent‑to‑latent diffusion trainer")
    parser.add_argument("-env", default="./config/environment.json", help="environment json file")
    parser.add_argument("-config", default="./config/config_train_16g.json", help="hyper‑parameter file")
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    parser.add_argument("--latent_dir", default="./z_latent_maps", help="folder with train/val/test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="./rflow_t1n_t2f", help="Directory to save models & logs")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load configs and set up deterministic env
    # ------------------------------------------------------------------
    with open(args.config) as f:
        cfg = json.load(f)
    with open(args.env) as f:
        env = json.load(f)

    seed_all()
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Instantiate UNet and scheduler
    # ------------------------------------------------------------------
    cfg["diffusion_def"]["in_channels"] = cfg["latent_channels"] * 3  # concat target+cond
    print(">> Instantiating UNet …")
    unet = instantiate(cfg["diffusion_def"], cfg).to(device)
    #unet_checkpoint = torch.load("./rflow_100/unet_latest.pt", map_location="cpu", weights_only=False)
    #unet.load_state_dict(unet_checkpoint.get("unet", unet_checkpoint), strict=True)
    #print("loaded checkpoint!")

    scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=True,
        sample_method="logit-normal",
        use_timestep_transform=True,
        base_img_size_numel=64 * 64 * 48,
        spatial_dim=3,
    )

    optimizer = AdamW(unet.parameters(), lr=cfg["diffusion_train"]["lr"], weight_decay=1e-4)
    scaler = GradScaler()

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    root = Path(args.latent_dir)
    train_loader = DataLoader(
        LatentPairDataset(root / "train"),
        batch_size=cfg["diffusion_train"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        LatentPairDataset(root / "val"),
        batch_size=cfg["diffusion_train"]["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Logging / bookkeeping setup
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "loss_history.csv"
    png_path = output_dir / "loss_curve.png"

    fieldnames = ["epoch", "step", "train_loss", "val_loss"]
    with csv_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    history: list[Dict[str, Any]] = []
    best_loss = float("inf")
    global_step = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    max_epochs = cfg["diffusion_train"]["max_epochs"]
    val_interval = cfg["diffusion_train"]["val_interval"]

    for epoch in range(max_epochs):
        # ------------------------ TRAIN ---------------------------------
        unet.train()
        epoch_losses = []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in train_pbar:
            tgt = batch["target"].to(device)
            cond = batch["cond"].to(device)
            seg  = batch["seg"].to(device)
            noise = torch.randn_like(tgt)
            timesteps = scheduler.sample_timesteps(tgt)
            noisy_latents = scheduler.add_noise(tgt, noise, timesteps)
            model_in = torch.cat([noisy_latents, cond, seg], 1)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.float16):
                noise_pred = unet(model_in, timesteps)
                loss = F.l1_loss(noise_pred, (tgt - noise))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

        train_loss = float(np.mean(epoch_losses))
        train_row = {"epoch": epoch, "step": global_step, "train_loss": train_loss, "val_loss": None}
        history.append(train_row)
        with csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(train_row)

        # ----------------------- VALIDATION -----------------------------
        if (epoch + 1) % val_interval == 0:
            unet.eval()
            val_losses = []
            with torch.no_grad(), autocast(device_type=device.type, dtype=torch.float16):
                val_pbar = tqdm(val_loader, desc=f"Val {epoch}", leave=False)
                for batch in val_pbar:
                    tgt = batch["target"].to(device)
                    cond = batch["cond"].to(device)
                    seg  = batch["seg"].to(device)
                    noise = torch.randn_like(tgt)
                    timesteps = scheduler.sample_timesteps(tgt)
                    noisy = scheduler.add_noise(tgt, noise, timesteps)
                    pred = unet(torch.cat([noisy, cond, seg], 1), timesteps)
                    l1 = F.l1_loss(pred, (tgt - noise)).item()
                    val_losses.append(l1)
                    val_pbar.set_postfix(l1=f"{l1:.4f}")

            val_loss = float(np.mean(val_losses))
            val_row = {"epoch": epoch, "step": global_step, "train_loss": None, "val_loss": val_loss}
            history.append(val_row)
            with csv_path.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(val_row)

            print(f"\n>> Validation | epoch {epoch} | L1 {val_loss:.4f}")

            # ------- Checkpointing -------------------------------------
            latest_ckpt = output_dir / "unet_latest.pt"
            torch.save({"unet": unet.state_dict(), "epoch": epoch, "val_l1": val_loss}, latest_ckpt)

            if val_loss < best_loss:
                best_loss = val_loss
                best_ckpt = output_dir / "unet_best.pt"
                torch.save({"unet": unet.state_dict(), "epoch": epoch, "val_l1": val_loss}, best_ckpt)
                print(f"   ↓ new best model saved (L1 {val_loss:.4f})")

            if (epoch + 1) % 50 == 0:
                epoch_ckpt = output_dir / f"unet_epoch_{epoch}.pt"
                torch.save({"unet": unet.state_dict(), "epoch": epoch, "val_l1": val_loss}, epoch_ckpt)

            # ------- Plot loss curve -----------------------------------
            train_pts = [(h["step"], h["train_loss"]) for h in history if h["train_loss"] is not None]
            val_pts = [(h["step"], h["val_loss"]) for h in history if h["val_loss"] is not None]

            plt.figure(figsize=(6, 4))
            if train_pts:
                x, y = zip(*train_pts)
                plt.plot(x, y, label="train")
            if val_pts:
                x, y = zip(*val_pts)
                plt.plot(x, y, label="val")
            plt.xlabel("step")
            plt.ylabel("L1 (noise pred)")
            if train_pts and min(y) > 0:
                plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.title(f"Loss curve ({datetime.now().strftime('%m-%d %H:%M')})")
            plt.savefig(png_path)
            plt.close()

    print("Training finished! Checkpoints are in:", output_dir)

# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
