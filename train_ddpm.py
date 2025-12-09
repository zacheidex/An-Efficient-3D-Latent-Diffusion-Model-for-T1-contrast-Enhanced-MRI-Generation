# --- Imports (unchanged) ---
import argparse, json, math, os, random, importlib
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm.auto import tqdm
from monai.networks.schedulers import DDPMScheduler
from monai.inferers import DiffusionInferer
import csv
import matplotlib.pyplot as plt
from datetime import datetime

##### Helpers #########################################################################################

def _resolve(cfg: Dict[str, Any], root: Dict[str, Any]):
    if isinstance(cfg, dict):
        return {k: _resolve(v, root) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_resolve(v, root) for v in cfg]
    if isinstance(cfg, str):
        if cfg.startswith("@"):
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

##### Dataset #########################################################################################


class LatentPairDataset(Dataset):
    """
    Expects files of the form
        …-t1c_z_mu.pt      | …-t1c_z_sigma.pt
        …-t1n_z_mu.pt      | …-t1n_z_sigma.pt
    and returns a single z-sample for both target (t1c) and condition (t1n).
    """
    def __init__(self, split_dir: Path):
        self._pairs: list[tuple[Path, Path, Path, Path]] = []
        for mu_tgt in split_dir.rglob("*-t1c_z_mu.pt"):
            sig_tgt  = mu_tgt.with_name(mu_tgt.name.replace("_z_mu.pt", "_z_sigma.pt"))
            mu_cond  = Path(str(mu_tgt).replace("-t1c_z_mu.pt", "-t1n_z_mu.pt"))
            sig_cond = Path(str(mu_tgt).replace("-t1c_z_mu.pt", "-t1n_z_sigma.pt"))
            if all(p.exists() for p in (sig_tgt, mu_cond, sig_cond)):
                self._pairs.append((mu_tgt, sig_tgt, mu_cond, sig_cond))

        if not self._pairs:
            raise RuntimeError(f"No *-t1c_z_mu.pt files found under {split_dir}")

    def __len__(self): return len(self._pairs)


    def __getitem__(self, idx):
        mu_tgt, sig_tgt, mu_cond, sig_cond = self._pairs[idx]
        μ_t, σ_t = torch.load(mu_tgt),  torch.load(sig_tgt)
        μ_c, σ_c = torch.load(mu_cond), torch.load(sig_cond)

        eps  = torch.randn_like(μ_t)
        z_tgt  = μ_t + σ_t * eps

        eps  = torch.randn_like(μ_c)
        z_cond = μ_c + σ_c * eps

        return {
            "target": z_tgt,
            "cond":   z_cond,
        }


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

##### Main #########################################################################################

def main():
    parser = argparse.ArgumentParser(description="Latent‑to‑latent DDPM trainer")
    parser.add_argument("-config", default="config.json", help="training config json file")
    parser.add_argument("-g", "--gpus", default=1, type=int, help="num gpus")
    parser.add_argument("--latent_dir", default="./z_latent_maps", help="latent maps folder")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument("--output_dir", default="./DDPM_correct_scale_factor_epoch_225", help="Directory to save models & logs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    latent_channels = cfg["latent_channels"]
    cfg["diffusion_def"]["in_channels"] = latent_channels * 2  # target + condition

    device = torch.device(args.device)
    seed_all()

    # --- Models & scheduler ---
    unet = instantiate(cfg["diffusion_def"], cfg).to(device)
    unet_checkpoint = torch.load('./DDPM_correct_scale_factor_epoch_46/unet_latest.pt', map_location="cpu", weights_only=False)
    unet.load_state_dict(unet_checkpoint.get("unet", unet_checkpoint), strict=True)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0195,
        schedule="scaled_linear_beta",
        clip_sample=False
    )
    inferer = DiffusionInferer(noise_scheduler)
    optimizer = AdamW(unet.parameters(), lr=cfg["diffusion_train"]["lr"], weight_decay=1e-4)
    scaler = GradScaler()

    # --- Dataloaders ---
    print(">> Preparing datasets…")
    root = Path(args.latent_dir)
    train_ds = LatentPairDataset(root / "train")
    val_ds   = LatentPairDataset(root / "val")
    train_loader = DataLoader(train_ds, batch_size=cfg["diffusion_train"]["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["diffusion_train"]["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # --- Training loop ---
    max_epochs   = cfg["diffusion_train"]["max_epochs"]
    val_interval = cfg["diffusion_train"]["val_interval"]
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "loss_history.csv"
    png_path = save_dir / "loss_curve.png"
    history, best_loss = [], float("inf")
    global_step = 0

    bucket_edges  = torch.tensor([0, 10, 20, 50, 100, 250, 500, 1000], device=device)
    n_buckets     = len(bucket_edges) - 1


    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "train_loss", "val_loss"])
        writer.writeheader()

    for epoch in range(max_epochs):
        unet.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in train_pbar:
            global_step += 1
            tgt  = batch["target"].to(device)
            cond = batch["cond"].to(device)
            noise = torch.randn_like(tgt)

            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (tgt.size(0),), device=device).long()
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.float16):
                noise_pred = inferer(
                    noise = noise,
                    inputs = tgt,
                    condition=cond,
                    mode="concat",
                    diffusion_model=unet,
                    timesteps=timesteps,
                )
                loss = F.mse_loss(noise_pred, noise)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
            train_loss_val = loss.item()
            history.append({"epoch": epoch, "step": global_step, "train_loss": train_loss_val, "val_loss": None})

        # --- Validation ---
# --- Validation --------------------------------------------------------------
        if (epoch + 1) % val_interval == 0:
            unet.eval()

            # fresh accumulators just for this validation pass
            bucket_losses = torch.zeros(n_buckets, device=device)
            bucket_counts = torch.zeros_like(bucket_losses)
            val_loss = []

            with torch.no_grad(), autocast(device_type=device.type, dtype=torch.float16):
                val_pbar = tqdm(val_loader, desc=f"Val {epoch}", leave=False)
                for batch in val_pbar:
                    tgt  = batch["target"].to(device)
                    cond = batch["cond"].to(device)
                    noise = torch.randn_like(tgt)
                    timesteps = torch.randint(
                        0, noise_scheduler.num_train_timesteps, (tgt.size(0),),
                        device=device).long()

                    noise_pred = inferer(
                        noise=noise,
                        inputs=tgt,
                        condition=cond,
                        mode="concat",
                        diffusion_model=unet,
                        timesteps=timesteps,
                    )

                    # --- per-sample MSE (not averaged over batch!) --------------------
                    mse = F.mse_loss(noise_pred, noise, reduction="none")
                    mse = mse.flatten(1).mean(1)           # shape (B,)

                    # accumulate into buckets
                    for i in range(n_buckets):
                        m = (timesteps >= bucket_edges[i]) & (timesteps < bucket_edges[i+1])
                        if m.any():
                            bucket_losses[i] += mse[m].sum()
                            bucket_counts[i] += m.sum()

                    val_loss.extend(mse.tolist())
                    val_pbar.set_postfix(mse=f"{mse.mean().item():.4f}")

            # --------- print bucket report -------------------------------------------
            avg = bucket_losses / bucket_counts.clamp(min=1)
            print("\nloss by t-bucket:",
                  " | ".join(f"{bucket_edges[i]:>3}-{bucket_edges[i+1]-1:<3}: {avg[i]:.4f}"
                             for i in range(n_buckets)))


            vl = np.mean(val_loss)
            history.append({"epoch": epoch, "step": global_step, "train_loss": None, "val_loss": vl})
            print(f">> Validation | epoch {epoch} | MSE {vl:.4f}")
            latest_ckpt = save_dir / "unet_latest.pt"
            torch.save({"unet": unet.state_dict(), "epoch": epoch, "val_mse": vl}, latest_ckpt)
            if vl < best_loss:
                best_loss = vl
                best_ckpt = save_dir / "unet_best.pt"
                torch.save({"unet": unet.state_dict(), "epoch": epoch, "val_mse": vl}, best_ckpt)
                print(f"   ↓ new best model saved (MSE {vl:.4f})")

            if epoch % 50 == 0:
                epoch_ckpt = save_dir / f"unet_epoch_{epoch}.pt"
                torch.save({"unet": unet.state_dict(), "epoch": epoch, "val_mse": vl}, epoch_ckpt)

            with csv_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "step", "train_loss", "val_loss"])
                writer.writerows(history[-2:])
            # Plot
            train_pts = [(h["step"], h["train_loss"]) for h in history if h["train_loss"] is not None]
            val_pts   = [(h["step"], h["val_loss"])   for h in history if h["val_loss"]   is not None]
            plt.figure(figsize=(6,4))
            if train_pts:
                x, y = zip(*train_pts)
                plt.plot(x, y, label="train")
            if val_pts:
                x, y = zip(*val_pts)
                plt.plot(x, y, label="val")
            plt.xlabel("step"); plt.ylabel("MSE")
            if train_pts and min(y) > 0:
                plt.yscale("log")
            plt.legend(); plt.tight_layout()
            plt.title(f"Loss curve ({datetime.now().strftime('%m-%d %H:%M')})")
            plt.savefig(png_path)
            plt.close()

    print("Training finished! Best models are in:", save_dir)


if __name__ == "__main__":
    main()
