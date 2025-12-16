# Quickstart

## 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> **PyTorch note:** install the right CUDA build for your machine from pytorch.org, then install the rest.

## 2) Put data in place

Create the preprocessed folder layout described in `docs/DATA_LAYOUT.md`.

## 3) Precompute latents

```bash
python generate_latent_maps.py \
  --data-root ./data/BraTS_preprocessed \
  --out-dir   ./data/z_latents \
  --modalities t1n,t2f,t1c \
  --ckpt ./checkpoints/autoencoder_epoch273.pt \
  --device cuda:0
```

## 4) Train RFlow in latent space

```bash
python train_rflow.py \
  --latent_dir ./data/z_latents \
  --config ./config/config_train_16g.json \
  --output_dir ./runs/rflow \
  --device cuda
```

## 5) Run inference (latent → T1c)

```bash
python test_rflow.py \
  --latent-root ./data/z_latents \
  --input-root ./data/BraTS_preprocessed \
  --output-dir ./outputs/rflow \
  --config-unet ./config/config_train_16g.json \
  --ckpt ./checkpoints/autoencoder_epoch273.pt \
  --unet-ckpt ./runs/rflow/unet_best.pt
```
