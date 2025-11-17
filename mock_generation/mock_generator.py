import numpy as np
from astropy.table import Table
import os
from tqdm import tqdm
import time
import healpy as hp


def astrometry_sim(n_objects, mask=None, anisotropy_map=None,
                   threads_var=1, nside_new_var=67108864, nside_regions2sim=64):
    """
    Generate isotropic or anisotropic sky positions as Cartesian unit vectors.

    Parameters
    ----------
    n_objects : int
        Number of simulated objects (Cartesian vectors).
    mask : np.ndarray, optional
        HEALPix mask (same nside as `anisotropy_map`). Non-zero pixels define
        the region where objects are generated (NESTED scheme is assumed).
    anisotropy_map : np.ndarray, optional
        HEALPix map that sets the relative probability density on each sky
        direction (same nside as `mask`). Values are re-normalized internally.
    threads_var : int
        Number of threads used in the (original) aberration code (kept for API
        compatibility; not used explicitly here).
    nside_new_var : int
        HEALPix NSIDE used to randomly sample object positions (subpixels).
    nside_regions2sim : int
        When not using a mask, this would set the NSIDE of the base pixels.
        (Kept for API compatibility; does not affect the distribution here.)

    Returns
    -------
    astrometry : np.ndarray, shape (N, 3)
        Array of Cartesian unit vectors (x, y, z) for the simulated directions.
    """
    # Get base pixels where mask is non-zero
    pix2sim = np.argwhere(mask != 0)
    npix = pix2sim.size
    nside_old = hp.npix2nside(mask.size)
    nside_new = nside_new_var
    # Factor used to choose subpixels of a base pixel (NESTED scheme)
    npix_coef = int((nside_new / nside_old) ** 2)

    anisotropy_map = anisotropy_map * mask
    anisotropy_map = anisotropy_map[anisotropy_map != 0]
    mean_count = int(n_objects / npix)
    anisotropy_map = (1 + anisotropy_map) * mean_count
    anisotropy_map = anisotropy_map / np.sum(anisotropy_map)

    # Randomly pick base pixels according to anisotropy_map
    random_pix = np.random.RandomState().choice(
        pix2sim.flatten(), size=n_objects, p=anisotropy_map
    )
    random_pix = np.array(random_pix * npix_coef)

    # Random choice of subpixel within each base pixel
    random_pix_new_idx = np.random.RandomState().randint(
        0, npix_coef + 1, size=n_objects
    )
    random_pix_new_idx = random_pix + random_pix_new_idx
    random_pix = None

    # Convert pixel indices to Cartesian vectors
    astrometry = np.array(
        hp.pix2vec(nside_new, random_pix_new_idx, nest=True)
    ).T

    return astrometry


def direcoes_g(n_gal, mask_var, anisotropy_map_var, n_threads=16):
    """
    Generate random tangent-unit vectors g for each galaxy direction n.

    The function returns:
      - n: line-of-sight unit vectors (N, 3)
      - g: random tangent-unit vectors on the local tangent plane at n,
           corresponding to a random position angle.

    Parameters
    ----------
    n_gal : int
        Number of galaxies (directions) to simulate.
    mask_var : np.ndarray
        HEALPix mask used to define where galaxies are simulated.
    anisotropy_map_var : np.ndarray
        HEALPix map encoding the anisotropy of the source distribution.
    n_threads : int
        Number of threads passed to `astrometry_sim` (kept for API compatibility).

    Returns
    -------
    n : np.ndarray, shape (N, 3)
        Line-of-sight unit vectors.
    g : np.ndarray, shape (N, 3)
        Tangent-unit vectors (major-axis directions) on the sphere.
    """
    n = astrometry_sim(
        n_gal,
        mask=mask_var,
        anisotropy_map=anisotropy_map_var,
        threads_var=n_threads,
        nside_new_var=67108864,
        nside_regions2sim=64,
    )

    # Use hp.pixelfunc.vec2ang to obtain longitude and latitude [deg] from n
    lon, lat = hp.pixelfunc.vec2ang(n, lonlat=True)
    lon, lat = np.array(lon, dtype=np.float64), np.array(lat, dtype=np.float64)

    # Convert to radians
    ra = np.radians(lon)
    dec = np.radians(lat)

    # Tangent-plane basis vectors at each (ra, dec)
    delta = np.array([
        -np.sin(dec) * np.cos(ra),
        -np.sin(dec) * np.sin(ra),
        np.cos(dec)
    ])

    alpha = np.array([
        -np.sin(ra),
        np.cos(ra),
        np.zeros(ra.size)
    ])
    alpha = alpha.T
    delta = delta.T

    # Draw random position angles θ uniformly in [0, 2π)
    theta = np.random.uniform(0, 2 * np.pi, size=n_gal)

    # Build g in the local tangent basis (α-hat, δ-hat)
    g = np.sin(theta)[:, np.newaxis] * delta - np.cos(theta)[:, np.newaxis] * alpha

    return n, g


def rotate_tangent_vectors(g, n, d, theta, theta_error_map, nest=True, rng=None):
    """
    Rotate tangent vectors toward a reference direction and then apply
    measurement errors drawn from a HEALPix map.

    Steps
    -----
    1) Rotate the tangent vectors `g` toward the projection of `d` onto the
       tangent plane defined by `n`, saturating at the maximum angle `theta`.
    2) After saturation, apply a measurement error: each g_rot is rotated
       around `n` by a random angle drawn from N(0, σ), where σ is read from
       a HEALPix map (`theta_error_map`).

    Parameters
    ----------
    g : np.ndarray, shape (N, 3)
        Original tangent-unit vectors.
    n : np.ndarray, shape (N, 3)
        Line-of-sight unit vectors (normals to the tangent planes).
    d : np.ndarray, shape (3,)
        Reference direction vector.
    theta : float
        Maximum rotation (saturation) angle, in radians.
    theta_error_map : np.ndarray
        HEALPix map with 1σ errors (radians) as a function of direction.
    nest : bool
        If True, assume NESTED ordering for the error map.
    rng : np.random.Generator or None
        Random generator for reproducibility. If None, a new default generator
        is created.

    Returns
    -------
    g_err : np.ndarray, shape (N, 3)
        Tangent-unit vectors after saturation + error rotation on the tangent plane.
    """
    # ------------------------------------------------------------------
    # --- 0) Contamination definition: which axis is rotated ----------
    # ------------------------------------------------------------------
    N = g.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    # Working copy of g (by default, the major axis)
    g_work = g.copy()

    # ------------------------------------------------------------------
    # --- 1) Original rotation with saturation (logic unchanged) -------
    # ------------------------------------------------------------------
    dot_dn = np.einsum('ij,j->i', n, d)
    d_proj = d - dot_dn[:, None] * n
    norm_d_proj = np.linalg.norm(d_proj, axis=1)
    eps = 1e-10
    valid = norm_d_proj > eps

    t_target = np.empty_like(g_work)
    t_target[valid] = d_proj[valid] / norm_d_proj[valid, None]
    t_target[~valid] = g_work[~valid]

    dot_gt = np.einsum('ij,ij->i', g_work, t_target)
    t_target[dot_gt < 0] *= -1

    dot_gt = np.clip(np.einsum('ij,ij->i', g_work, t_target), -1, 1)
    angle = np.arccos(dot_gt)

    cross_gt = np.cross(g_work, t_target)
    sign = np.sign(np.einsum('ij,ij->i', n, cross_gt))
    sign[sign == 0] = 1.0

    delta = np.minimum(angle, theta)
    phi = sign * delta

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    n_cross_g = np.cross(n, g_work)

    g_rot = g_work * cos_phi[:, None] + n_cross_g * sin_phi[:, None]
    saturate = angle < theta
    g_rot[saturate] = t_target[saturate]

    # ------------------------------------------------------------------
    # --- 2) Measurement error via HEALPix map (logic unchanged) -------
    # ------------------------------------------------------------------
    nside = hp.npix2nside(theta_error_map.size)
    pix_idx = hp.vec2pix(nside, n[:, 0], n[:, 1], n[:, 2], nest=nest)
    sigma = theta_error_map[pix_idx]
    theta_err = rng.normal(0.0, sigma, size=N)

    cos_e = np.cos(theta_err)
    sin_e = np.sin(theta_err)
    cross_ng = np.cross(n, g_rot)
    g_err = g_rot * cos_e[:, None] + cross_ng * sin_e[:, None]

    # Renormalize to unit length
    g_err /= np.linalg.norm(g_err, axis=1)[:, None]

    return g_err


def rotate_tangent_vectors_no_error(g, n, d, theta, nest=True, rng=None):
    """
    Rotate tangent vectors toward a reference direction without applying
    any measurement errors.

    Steps
    -----
    1) Rotate the tangent vectors `g` toward the projection of `d` onto the
       tangent plane defined by `n`, saturating at the maximum angle `theta`.
    2) No additional random error is applied (pure IA-like rotation only).

    Parameters
    ----------
    g : np.ndarray, shape (N, 3)
        Original tangent-unit vectors.
    n : np.ndarray, shape (N, 3)
        Line-of-sight unit vectors (normals to the tangent planes).
    d : np.ndarray, shape (3,)
        Reference direction vector.
    theta : float
        Maximum rotation (saturation) angle, in radians.
    nest : bool
        Kept for interface consistency (no map is used here).
    rng : np.random.Generator or None
        Random generator for reproducibility (not used explicitly).

    Returns
    -------
    g_rot : np.ndarray, shape (N, 3)
        Tangent-unit vectors after saturation (no error applied).
    """
    # ------------------------------------------------------------------
    # --- 0) Contamination definition: which axis is rotated ----------
    # ------------------------------------------------------------------
    N = g.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    # Working copy of g (by default, the major axis)
    g_work = g.copy()

    # ------------------------------------------------------------------
    # --- 1) Original rotation with saturation (logic unchanged) -------
    # ------------------------------------------------------------------
    dot_dn = np.einsum('ij,j->i', n, d)
    d_proj = d - dot_dn[:, None] * n
    norm_d_proj = np.linalg.norm(d_proj, axis=1)
    eps = 1e-10
    valid = norm_d_proj > eps

    t_target = np.empty_like(g_work)
    t_target[valid] = d_proj[valid] / norm_d_proj[valid, None]
    t_target[~valid] = g_work[~valid]

    dot_gt = np.einsum('ij,ij->i', g_work, t_target)
    t_target[dot_gt < 0] *= -1

    dot_gt = np.clip(np.einsum('ij,ij->i', g_work, t_target), -1, 1)
    angle = np.arccos(dot_gt)

    cross_gt = np.cross(g_work, t_target)
    sign = np.sign(np.einsum('ij,ij->i', n, cross_gt))
    sign[sign == 0] = 1.0

    delta = np.minimum(angle, theta)
    phi = sign * delta

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    n_cross_g = np.cross(n, g_work)

    g_rot = g_work * cos_phi[:, None] + n_cross_g * sin_phi[:, None]
    saturate = angle < theta
    g_rot[saturate] = t_target[saturate]

    return g_rot


def _random_unit_vector(rng):
    """Draw an isotropic unit vector in R^3."""
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


# ------------------------ PLACEHOLDER INPUT FILES ------------------------ #
# Replace these paths with the actual locations of your maps.
anisotropy = hp.read_map("path/to/disc_anisotropy_nside64.fits", nest=True)
mask_64 = hp.read_map("path/to/disc_mask_nside64.fits", nest=True)
error_map = hp.read_map("path/to/disc_error_nside64.fits", nest=True)

# Number of galaxies in the observed sample (including contamination)
n_gal = 2999517  # e.g. considering contamination 3,007,638
nsims = 500

# Measured IA direction (HEALPix NSIDE=32, pixel index 3793, NESTED)
d_measured = hp.pix2vec(32, 3793, nest=True)
theta_measured = np.deg2rad(7.289582102 / 60)  # IA amplitude in radians (from data)

# Lensing-like extra rotation: 1 arcmin in radians
phi_lens_rad = np.deg2rad(1 / 60)
theta_lens = phi_lens_rad  # radians

# Output directory (placeholder — change to your preferred path)
output_dir = "path/to/dd_obs_meas_sims"
os.makedirs(output_dir, exist_ok=True)

# Timer for the full simulation set
start_time = time.time()

# For each realization, draw a random lensing direction and simulate one map
for j in tqdm(range(nsims)):
    # Generate intrinsic galaxy directions (n) and random tangent axes (g)
    n, g = direcoes_g(
        n_gal,
        mask_var=mask_64,
        anisotropy_map_var=anisotropy
    )

    rng = np.random.default_rng(j + 10000)

    # Random lensing direction per simulation
    d_random = _random_unit_vector(rng)

    # Apply IA-like rotation toward the measured direction (no error)
    g_ia = rotate_tangent_vectors_no_error(
        g, n, d_measured, theta_measured, nest=True, rng=rng
    )

    # Apply lensing-like rotation + measurement error from the map
    g_lens = rotate_tangent_vectors(
        g_ia, n, d_random, theta_lens, error_map, nest=True, rng=rng
    )

    # Split components for saving
    n_x, n_y, n_z = n[:, 0], n[:, 1], n[:, 2]
    g_rot_x, g_rot_y, g_rot_z = g_lens[:, 0], g_lens[:, 1], g_lens[:, 2]

    # Build the output table
    simulation_table = Table(
        [n_x, n_y, n_z, g_rot_x, g_rot_y, g_rot_z],
        names=("n_x", "n_y", "n_z", "g_rot_x", "g_rot_y", "g_rot_z"),
    )

    # Output filename for this realization
    output_path = os.path.join(output_dir, f"dd_simulation_obs_{j}.fits")

    # Write the FITS table
    simulation_table.write(output_path, overwrite=True)

