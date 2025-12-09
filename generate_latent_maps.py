#!/usr/bin/env python
# inference_vae.py – run VAE forward pass + metrics (fp16-friendly, robust, with sliding window option)
import argparse, json, warnings, numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import csv
import re
import torch
from monai.config import print_config
from monai.data import Dataset, DataLoader, NibabelWriter
from monai.metrics import PSNRMetric, SSIMMetric
from monai.utils import set_determinism
from monai.inferers import SlidingWindowInferer

from maisi.scripts.transforms import VAE_Transform
from maisi.scripts.utils import define_instance
from collections import defaultdict

print_config(); warnings.filterwarnings("ignore")

# ─────────────── CLI ───────────────
cli = argparse.ArgumentParser()
cli.add_argument("--ckpt",       default='autoencoder_epoch273.pt', help="checkpoint .pt/.pth")
cli.add_argument("--data-root",  default='/data/zeidex/BraTS_preprocessed', help="BraTS_preprocessed dir")
cli.add_argument("--out-dir",    default="./z_latent_maps_seg")
cli.add_argument("--batch",      type=int, default=1)
cli.add_argument("--device",     default="cuda:0")
cli.add_argument("--save-latents", default=True)
cli.add_argument("--save-nifti", default=True)
cli.add_argument("--sliding-window", action="store_true", help="Use sliding window for inference")
cli.add_argument("--sw-overlap", type=float, default=0.5, help="Sliding window overlap (default: 0.5)")
args = cli.parse_args()

# ─────────────── Device, reproducibility ───────────────
device  = torch.device(args.device)
out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
set_determinism(0)

# ─────────────── Load configs ───────────────
env_json   = "./maisi/configs/environment_maisi_vae_train.json"
arch_json  = "./maisi/configs/config_maisi3d-rflow.json"
train_json = "./maisi/configs/config_maisi_vae_train.json"

env_args  = argparse.Namespace(**json.load(open(env_json)))
arch_dict = json.load(open(arch_json))
for k, v in arch_dict.items():
    setattr(env_args, k, v)
train_dict = json.load(open(train_json))
auto_train = train_dict["autoencoder_train"]
data_opts  = train_dict["data_option"]
print(f"Loaded architecture: {arch_json}\nLoaded loader hyper-params: {train_json}")

# ─────────────── Save NIfTI ───────────────
def save_nibabel(pred_tensor, meta, file_path):
    arr = pred_tensor.detach().cpu().float().numpy()
    ch_dim = 0 if arr.ndim == 4 and arr.shape[0] == 1 else None
    writer = NibabelWriter()
    writer.set_data_array(arr, channel_dim=ch_dim)
    writer.set_metadata({
        "affine": meta.get("affine", np.eye(4)),
        "original_affine": meta.get("original_affine", np.eye(4)),
    })
    writer.write(str(file_path), verbose=False)

# ─────────────── Build test list ───────────────
def collect(split_dir: Path, split_name, suffixes=("seg.nii.gz",)):
    cases = []
    for case in sorted(split_dir.iterdir()):
        if not case.is_dir():
            continue
        for suf in suffixes:
            for imgfile in sorted(case.glob(f"*{suf}")):
                cases.append({
                    "input":  str(imgfile),
                    "target": str(imgfile),
                    "cid":    case.name,
                    "filename": imgfile.name,
                    "class":  "mri",
                    "split":  split_name
                })
    return cases

test_list = (
    collect(Path(args.data_root) / "test", "test") +
    collect(Path(args.data_root) / "train", "train") +
    collect(Path(args.data_root) / "val", "val")
)


# ─────────────── Robust full-volume transform ───────────────
# This matches typical MONAI transform idioms: load, reorient, spacing, scale.
vt = VAE_Transform(
    is_train=False,
    random_aug=None,                # No augmentation for inference!
    k=1,                            # Not used
    val_patch_size=None,            # NO PATCHING for inference
    output_dtype=torch.float16,
    spacing_type=data_opts["spacing_type"],
    spacing=data_opts["spacing"],
    image_keys=["input", "target"],
    select_channel=data_opts["select_channel"],
)

ds_test = Dataset(test_list, vt)    # CacheDataset not necessary for small test sets
dl_test = DataLoader(ds_test, batch_size=args.batch, num_workers=8)  # Always batch 1 for 3D full volumes

# ─────────────── Load model ───────────────
env_args.autoencoder_def["num_splits"] = 1
ae = define_instance(env_args, "autoencoder_def").to(device).half()
ckpt = torch.load(args.ckpt, map_location="cpu")
ae.load_state_dict(ckpt.get("autoencoder", ckpt), strict=False)
ae.eval(); print("✅  checkpoint loaded (fp16 mode).")

# ─────────────── Sliding Window Inferer setup ───────────────
sw_roi_size = (256, 256, 192)
inferer = None
if args.sliding_window:
    inferer = SlidingWindowInferer(
        roi_size=sw_roi_size,
        sw_batch_size=1,
        overlap=args.sw_overlap,
        mode="gaussian"
    )
    print(f"Sliding window inference enabled. ROI size: {sw_roi_size}, overlap: {args.sw_overlap}")

# ─────────────── Metrics ───────────────
psnr_metric = PSNRMetric(max_val=1.0)
ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)
psnr_total = ssim_total = nmse_num = nmse_den = 0.0
sample_count = 0

metrics_csv_path = out_dir / "inference_metrics_seg.csv"
new_file         = not metrics_csv_path.exists()
metrics_file     = open(metrics_csv_path, "a", newline="")
metrics_writer   = csv.writer(metrics_file)
if new_file:
    metrics_writer.writerow(
        ["case_id", "filename", "psnr", "ssim", "nmse", "latent_path"]
    )

all_psnr = []
all_ssim = []
all_nmse = []
all_latent_paths = []

# For per-modality summary
per_modality_psnr = defaultdict(list)
per_modality_ssim = defaultdict(list)
per_modality_nmse = defaultdict(list)

def minmax_normalize(tensor):
    # Works for a batch of images: (B, C, D, H, W) or (B, D, H, W)
    return (tensor - tensor.amin(dim=[*range(1, tensor.ndim)], keepdim=True)) / (
        tensor.amax(dim=[*range(1, tensor.ndim)], keepdim=True) - tensor.amin(dim=[*range(1, tensor.ndim)], keepdim=True) + 1e-8
    )


with torch.no_grad():
    for idx, batch in enumerate(tqdm(dl_test, desc="Inference", unit="vol")):
        x = batch["input"].to(device).half()
        x = minmax_normalize(x)
        y = batch["target"].to(device).half()
        y = minmax_normalize(y)

        batch_size = x.shape[0]

        if args.sliding_window:
            pred = inferer(x, ae)
            z_mu = torch.zeros_like(pred)  # Replace with actual latent inference if needed
            z_sigma = torch.zeros_like(pred)
        else:
            pred, z_mu, z_sigma = ae(x)

        #z = z_mu + z_sigma*torch.randn_like(z_sigma) this is how to calculate z to train the LDM

        for i in range(batch_size):
            meta = batch["input_meta_dict"][i] if "input_meta_dict" in batch else {}
            entry = test_list[idx * args.batch + i]
            cid = entry["cid"]
            filename = entry["filename"]
            split = entry["split"]  # This is now available!

            case_dir = out_dir / split / cid
            case_dir.mkdir(parents=True, exist_ok=True)

            filename_no_ext = re.sub(r"\.nii(\.gz)?$", "", filename)
            single_pred = pred[i].unsqueeze(0).float()
            single_gt = y[i].unsqueeze(0).float()

            psnr_i = psnr_metric(single_pred, single_gt).mean().item()
            ssim_i = ssim_metric(single_pred, single_gt).mean().item()
            nmse_i = ((single_pred - single_gt) ** 2).sum().item() / ((single_gt) ** 2).sum().item()

            # Determine modality
            if filename.endswith('-t1c.nii.gz'):
                modality = 't1c'
            elif filename.endswith('t2f.nii.gz'):
                modality = 't2f'
            elif filename.endswith('t1n.nii.gz'):
                modality = 't1n'
            elif filename.endswith('seg.nii.gz'):
                modality = 'seg'
            else:
                modality = 'unknown'

            per_modality_psnr[modality].append(psnr_i)
            per_modality_ssim[modality].append(ssim_i)
            per_modality_nmse[modality].append(nmse_i)

            print(f"Case: {cid} | File: {filename} | PSNR: {psnr_i:.2f} | SSIM: {ssim_i:.4f} | NMSE: {nmse_i:.6f}")

            result_name = filename.replace('.nii.gz', '_pred.nii.gz')

            if args.save_nifti:
                save_nibabel(single_pred[0], meta, case_dir / result_name)

            latent_path = ""
            if args.save_latents:
                latent_path = str(case_dir / (filename_no_ext + "_latent.pt"))
                torch.save(z_mu[i].cpu(),    case_dir / f"{filename_no_ext}_z_mu.pt")
                torch.save(z_sigma[i].cpu(), case_dir / f"{filename_no_ext}_z_sigma.pt")

            # Save to CSV per-case row
            metrics_writer.writerow([cid, filename, psnr_i, ssim_i, nmse_i, latent_path])

            all_psnr.append(psnr_i)
            all_ssim.append(ssim_i)
            all_nmse.append(nmse_i)
            all_latent_paths.append(latent_path)

metrics_file.flush()  # ensure all per-case rows written

# --- Compute and save global summary ---
avg_psnr, std_psnr = np.mean(all_psnr), np.std(all_psnr)
avg_ssim, std_ssim = np.mean(all_ssim), np.std(all_ssim)
avg_nmse, std_nmse = np.mean(all_nmse), np.std(all_nmse)

metrics_writer.writerow([])
metrics_writer.writerow(["average", "", avg_psnr, avg_ssim, avg_nmse, ""])
metrics_writer.writerow(["std", "", std_psnr, std_ssim, std_nmse, ""])

# --- Compute and save per-modality summary ---
metrics_writer.writerow([])
metrics_writer.writerow(["==== Summary per modality ===="])
metrics_writer.writerow(["modality", "avg_psnr", "std_psnr", "avg_ssim", "std_ssim", "avg_nmse", "std_nmse"])

for modality in ["t1c", "t2f", "t1n","seg"]:
    psnr_vals = per_modality_psnr.get(modality, [])
    ssim_vals = per_modality_ssim.get(modality, [])
    nmse_vals = per_modality_nmse.get(modality, [])
    if psnr_vals:
        avg_psnr, std_psnr = np.mean(psnr_vals), np.std(psnr_vals)
        avg_ssim, std_ssim = np.mean(ssim_vals), np.std(ssim_vals)
        avg_nmse, std_nmse = np.mean(nmse_vals), np.std(nmse_vals)
        metrics_writer.writerow([
            modality, avg_psnr, std_psnr, avg_ssim, std_ssim, avg_nmse, std_nmse
        ])
    else:
        metrics_writer.writerow([modality, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])

metrics_file.close()

print("\n=== Test-set quality ===")
print(f"PSNR : {avg_psnr:.2f} ± {std_psnr:.2f} dB")
print(f"SSIM : {avg_ssim:.4f} ± {std_ssim:.4f}")
print(f"NMSE : {avg_nmse:.6f} ± {std_nmse:.6f}")
print("Results saved to:", out_dir.resolve())
if args.save_latents:
    print("Latent μ/σ maps saved alongside predictions.")