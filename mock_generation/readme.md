# LAIA Mock Generator 

This script generates mock catalogs of galaxy orientation vectors for testing the
large-scale axial intrinsic alignment (LAIA) estimator used in

> **“Where Galaxies Point: First Measurement of the Large-Scale Axial Intrinsic Alignment”**  
> Pedro da Silveira Ferreira, *et al.* – [arXiv:2511.10005](https://arxiv.org/abs/2511.10005)

It produces simulated catalogs with:

- A **measured IA-like rotation** toward a fixed direction on the sky; and  
- An additional **weak-lensing-like rotation** toward a *random* direction per realization,  
- Plus **position-angle measurement errors** drawn from a HEALPix error map.

The output FITS catalogs can be fed directly into the LAIA estimator code
(e.g. `run_estimator_new`) as mock “observed” samples.

---

## Requirements

Create a Python environment with the usual scientific stack:

```bash
conda create -n laia-mocks python=3.11
conda activate laia-mocks

pip install numpy astropy healpy tqdm
```

Dependencies:

- `numpy`
- `astropy`
- `healpy`
- `tqdm`
- (optional) `time`, `os` – part of the Python standard library

---

## Script overview

Assume the main file is called:

```text
dd_mock_generator.py
```

It contains the following key pieces:

### 1. `astrometry_sim(...)`

```python
astrometry = astrometry_sim(
    n_objects,
    mask=mask,
    anisotropy_map=anisotropy_map,
    threads_var=...,
    nside_new_var=...,
    nside_regions2sim=...
)
```

- Generates **line-of-sight unit vectors** on the sphere (`astrometry`, shape `(N, 3)`).
- Uses a **mask** and an **anisotropy map** (both HEALPix) to control where objects are drawn.
- Sampling is done at very high NSIDE (`nside_new_var`, default `67108864`, NESTED scheme),
  by picking subpixels inside base pixels selected according to `anisotropy_map`.

### 2. `direcoes_g(...)`

```python
n, g = direcoes_g(
    n_gal,
    mask_var=mask_64,
    anisotropy_map_var=anisotropy
)
```

- `n`: line-of-sight unit vectors (shape `(N, 3)`).
- `g`: random **tangent-unit vectors** (major-axis directions) on the local tangent plane.
- Internally:
  - Calls `astrometry_sim` to draw `n`.
  - Builds tangent basis vectors (`α-hat`, `δ-hat`).
  - Draws position angles θ uniformly in `[0, 2π)` and constructs `g`.

### 3. IA-like rotation (no error)

```python
g_ia = rotate_tangent_vectors_no_error(
    g, n, d_measured, theta_measured, nest=True, rng=rng
)
```

- Rotates each tangent vector `g` **toward a fixed direction** `d_measured`
  (the measured LAIA direction) by an angle up to `theta_measured` (saturation).
- Models **intrinsic alignment** only (no measurement error here).

### 4. Lensing-like rotation + measurement error

```python
g_lens = rotate_tangent_vectors(
    g_ia, n, d_random, theta_lens, error_map, nest=True, rng=rng
)
```

- For each realization:
  - Draw a **random lensing direction** `d_random` on the sphere.
  - Rotate `g_ia` toward `d_random` by a small angle `theta_lens`
    (e.g. 1 arcmin, weak-lensing-like).
  - Add a **position-angle error**: rotate around `n` by a Gaussian random angle
    `N(0, σ)`, where `σ` is read from a HEALPix **error map** (`error_map`).

The result is `g_lens`, which represents **IA + lensing + measurement error**.

### 5. Output FITS catalogs

For each realization `j`, the script writes:

```text
dd_simulation_obs_{j}.fits
```

with columns:

- `n_x`, `n_y`, `n_z` – components of the line-of-sight unit vector `n`
- `g_rot_x`, `g_rot_y`, `g_rot_z` – components of the final tangent-unit vector `g_lens`

This format is compatible with the LAIA estimator (`g_rot_*` and `n_*` columns).

---

## Input maps and placeholders

At the bottom of the script you will find the following placeholders:

```python
anisotropy = hp.read_map("path/to/disc_anisotropy_nside64.fits", nest=True)
mask_64    = hp.read_map("path/to/disc_mask_nside64.fits",       nest=True)
error_map  = hp.read_map("path/to/disc_error_nside64.fits",      nest=True)
```

You must replace these with your actual HEALPix files:

- **`disc_anisotropy_nside64.fits`**  
  Relative overdensity of sources (or any dimensionless anisotropy weight) at NSIDE=64.

- **`disc_mask_nside64.fits`**  
  Mask map at NSIDE=64 (non-zero pixels define where galaxies can be placed).

- **`disc_error_nside64.fits`**  
  Map of **position-angle errors σ(θ, φ)** in radians, at some NSIDE.  
  (The script automatically infers `nside` from the map size.)

Also set the output directory:

```python
output_dir = "path/to/dd_obs_meas_sims"
os.makedirs(output_dir, exist_ok=True)
```

---

## Physical parameters

At the bottom of the script you will see:

```python
n_gal = 2999517       # number of galaxies in the mock (disc sample size)
nsims = 500           # number of realizations

d_measured = hp.pix2vec(32, 3793, nest=True)
theta_measured = np.deg2rad(7.289582102 / 60)   # IA amplitude from data (in radians)

phi_lens_rad = np.deg2rad(1 / 60)               # 1 arcmin in radians
theta_lens = phi_lens_rad
```

You can change:

- `n_gal` to match the size of your observed catalog.
- `nsims` to the number of realizations you want.
- `d_measured` to your preferred **IA direction** (any unit vector in 3D);
  here it is specified via `(NSIDE=32, pixel=3793, NESTED)`.
- `theta_measured` to your measured IA amplitude (in radians).  
- `theta_lens` to control the strength of the lensing-like rotation.

---

## How to run

After editing the file paths and parameters:

```bash
conda activate laia-mocks

python dd_mock_generator.py
```

You should see a progress bar from `tqdm`:

```text
100%|█████████████████████████████████████| 500/500 [..:..<.., ..it/s]
```

and a set of FITS files in `output_dir`:

```text
dd_simulation_obs_0.fits
dd_simulation_obs_1.fits
...
dd_simulation_obs_499.fits
```

Each of these can then be passed to the LAIA estimator pipeline as a mock
“observed” catalog.

---

## Using the mocks with the LAIA estimator

If you are using the estimator described in the main LAIA code:

```python
from astropy.io import fits
from laia_estimator import run_estimator_new   # example import

mock_path = "path/to/dd_obs_meas_sims/dd_simulation_obs_0.fits"

run_estimator_new(
    mock_path,
    error_map=None,      # per-galaxy PA error is already encoded geometrically
    nside=32,
    show_plot=False,
    output_dir="path/to/results_dd_mock",
    sigma_col="sigma",   # if you later add a per-galaxy sigma column
    half_sky=True,
    nest=True,
)
```

In the current version of this mock generator, **no `sigma` column** is written;
the measurement error is encoded geometrically in `g_rot`. If you want to use
a per-galaxy `sigma` column in the estimator, you can:

1. Add a constant or position-dependent `sigma` array in the mock generator, and  
2. Write it as an extra column to the FITS table.

---

## Reproducibility

- Each realization `j` uses its own random generator:

  ```python
  rng = np.random.default_rng(j + 10000)
  ```

- This means the sequence of mocks is reproducible given:
  - The same input maps,
  - The same `n_gal`, `nsims` and parameters,
  - And the same script version.

---

If you run into any issue hooking these mocks into the estimator or want to
adapt this for other samples (e.g. bulge-dominated galaxies), you can reuse
the same structure by changing the input maps and the physical parameters.
