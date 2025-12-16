# FAQ

## Where do I get `autoencoder_epoch273.pt`?
This repo expects a **pretrained 3D VAE** checkpoint compatible with the MAISI configs under `maisi/configs/`.
If you trained your own VAE, point `--ckpt` to it.

Because checkpoints are often large (and may have separate licensing), they are not included in this repo.

## My run crashes with shape / spacing errors
These scripts assume inputs are already in the **same voxel grid** used during training (e.g., consistent orientation/spacing and a fixed-ish volume size).
If your data doesn't match, preprocess it first.

## I don't have a 16GB GPU
Edit `config/config_train_16g.json`:
- reduce `diffusion_train.batch_size`
- reduce UNet `channels`
- reduce `num_res_blocks`
