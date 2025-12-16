# T1C-RFlow: Efficient 3D latent diffusion for T1-contrast enhanced MRI generation

This repository contains code to **synthesize T1-contrast enhanced MRI (T1c)** from routine MRI inputs (e.g., **T1n** and **T2-FLAIR**) using an **efficient 3D latent diffusion** approach.

At a high level:

1. A pretrained **3D VAE** (from the included `maisi/` configs) encodes full 3D volumes into a compact latent space.
2. An RFlow-trained **3D diffusion UNet** learns to map **(T1n, T2-FLAIR) latents → T1c latents**.
3. The VAE decodes predicted latents back to a 3D T1c volume.

> VAE checkpoint is downloadable here.

---

## Repository layout

- `generate_latent_maps.py` — precompute VAE latents (μ/σ) for each modality and split  
- `train_rflow.py` — train RFlow in latent space  
- `test_rflow.py` — run inference and write out predicted T1c volumes  
- `config/` — default training config files (added so the scripts run out-of-the-box)
- `docs/` — short guides for data layout and common issues
- `maisi/` — MAISI components used for the VAE and transforms

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**PyTorch + CUDA:** install the correct PyTorch build first (see pytorch.org), then install the rest.

---

## Quickstart

See **`docs/QUICKSTART.md`** for copy/paste commands.

---

## Data layout

See **`docs/DATA_LAYOUT.md`** for the expected folder structure and file naming.

---

## Notes for new users

- These scripts assume your volumes are already **preprocessed** into a consistent 3D grid (orientation/spacing/size).
- If you run out of GPU memory, reduce `diffusion_train.batch_size` and/or the UNet `channels` in `config/config_train_16g.json`.

---

## Citation

If you use this code in academic work, please cite the associated manuscript (add your BibTeX here).

---

## License

- The license for this codebase is defined by the top-level repository license (if present).
- The `maisi/` subtree includes its own licensing terms; see `maisi/LICENSE*`.
