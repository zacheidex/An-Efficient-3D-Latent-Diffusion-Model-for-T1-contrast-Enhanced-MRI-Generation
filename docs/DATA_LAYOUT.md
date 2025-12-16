# Data layout expected by the scripts

This repo assumes **preprocessed** BraTS-style folders with this structure:

```
data/BraTS_preprocessed/
  train/
    <CASE_ID>/
      <CASE_ID>-t1n.nii.gz
      <CASE_ID>-t2f.nii.gz
      <CASE_ID>-t1c.nii.gz
      # optional:
      <CASE_ID>-seg.nii.gz
  val/
    <CASE_ID>/
      ...
  test/
    <CASE_ID>/
      ...
```

## Latent layout (output of `generate_latent_maps.py`)

After running latent precomputation, you'll have:

```
data/z_latents/
  train/
    <CASE_ID>/
      <CASE_ID>-t1n_z_mu.pt
      <CASE_ID>-t1n_z_sigma.pt
      <CASE_ID>-t2f_z_mu.pt
      <CASE_ID>-t2f_z_sigma.pt
      <CASE_ID>-t1c_z_mu.pt
      <CASE_ID>-t1c_z_sigma.pt
  val/
    ...
  test/
    ...
```

The training scripts sample *z* as `z = μ + σ·ε` on-the-fly.
