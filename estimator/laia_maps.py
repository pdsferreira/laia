import os
import time
import glob
import healpy as hp
import numpy as np
from astropy.table import Table, vstack 
from astropy.io import fits
import matplotlib.pyplot as plt
import math
from scipy.optimize import brentq, newton
from astropy.coordinates import SkyCoord
import astropy.units as u


# ===================== analytical core =====================


def recover_alpha(m, c_bar, *, tol=1e-12, maxiter=4096, eps=1e-9):
    """
    Solve for α in the equation

        0.5 + (c̄/π) * (α + 0.5 * sin(2α)) = m

    with α ∈ [0, π/2].

    Parameters
    ----------
    m : float or array-like
        Measured value of the E-field peak (per pixel).
    c_bar : float
        Effective concentration parameter C̄.
    tol : float, optional
        Absolute and relative tolerance passed to the root-finder.
    maxiter : int, optional
        Maximum number of iterations for the root-finder.
    eps : float, optional
        Small safety margin used when clipping m to the allowed range.

    Returns
    -------
    alpha : float
        Solution α in radians, constrained to [0, π/2].
    """
    m_min = 0.5
    m_max = 0.5 + 0.5 * c_bar
    m_safe = np.clip(m, m_min + eps, m_max - eps)

    def f(a):
        return 0.5 + (c_bar / math.pi) * (a + 0.5 * math.sin(2 * a)) - m_safe

    try:
        # Robust bracketed root-finding on [0, π/2]
        return brentq(f, a=0.0, b=math.pi / 2,
                      xtol=tol, rtol=tol, maxiter=maxiter)
    except ValueError:
        # Fallback to Newton's method (should rarely be needed)
        def fp(a):
            return (c_bar / math.pi) * (1.0 + math.cos(2 * a))

        return newton(f, x0=0.0, fprime=fp, tol=tol, maxiter=maxiter)


# ===================== utilities =====================


def _pix_to_coords(nside, ipix, *, nest=True):
    """
    Convert a HEALPix pixel index to equatorial and Galactic coordinates.

    Parameters
    ----------
    nside : int
        HEALPix NSIDE parameter.
    ipix : int
        HEALPix pixel index.
    nest : bool, optional
        If True, use NESTED ordering; otherwise RING.

    Returns
    -------
    ra_deg : float
        Right ascension in degrees.
    dec_deg : float
        Declination in degrees.
    l_deg : float
        Galactic longitude in degrees.
    b_deg : float
        Galactic latitude in degrees.
    """
    theta, phi = hp.pix2ang(nside, int(ipix), nest=nest)
    ra = float(np.degrees(phi))
    dec = float(90.0 - np.degrees(theta))
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    return ra, dec, float(c.galactic.l.deg), float(c.galactic.b.deg)


def _support_mask(*arrs):
    """
    Build a boolean mask where at least one of the input arrays is non-zero.

    Parameters
    ----------
    *arrs : array-like
        Input arrays (broadcastable to the same shape).

    Returns
    -------
    mask : ndarray of bool
        Boolean mask with True where any array is non-zero.
    """
    m = np.zeros_like(arrs[0], dtype=bool)
    for A in arrs:
        A = np.asarray(A)
        m |= (A != 0)
    return m


def _read_cbar(stats_path):
    """
    Read C_BAR (and possibly other stats) from a text file.

    Parameters
    ----------
    stats_path : str
        Path to the stats text file produced by `run_estimator_new`.

    Returns
    -------
    c_bar : float
        The C_BAR value.
    vals : dict
        Dictionary with all key/value pairs read from the file.
    """
    vals = {}
    with open(stats_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                k = parts[0].upper()
                try:
                    vals[k] = float(parts[1])
                except ValueError:
                    pass
    if "C_BAR" not in vals:
        raise RuntimeError(f"C_BAR not found in {stats_path}")
    return vals["C_BAR"], vals


def _safe_sigma_floor(sigma):
    """
    Enforce a robust floor of 1 degree on sigma and handle NaN/negative/zero.

    Parameters
    ----------
    sigma : array-like
        Angular uncertainty in radians.

    Returns
    -------
    sigma_safe : ndarray
        Array with NaNs replaced by 1 deg, and all values >= 1 deg.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma[~np.isfinite(sigma)] = np.deg2rad(1.0)
    sigma = np.maximum(sigma, np.deg2rad(1.0))
    return sigma


def _resolve_baseline(baseline, x_array, mask):
    """
    Resolve a baseline specification into a float value.

    Parameters
    ----------
    baseline : str, float or None
        If string:
          - "mean": use the mean of x_array[mask].
          - any other string: interpreted as a float.
        If float: used directly.
        If None: baseline is set to 0.0.
    x_array : array-like
        Array from which the baseline may be computed.
    mask : array-like of bool
        Boolean mask specifying which entries to use (for "mean").

    Returns
    -------
    baseline_value : float
        Resolved baseline value.
    """
    if isinstance(baseline, str):
        b = baseline.strip().lower()
        if b == "mean":
            return float(np.asarray(x_array)[mask].mean())
        else:
            return float(b)
    elif baseline is None:
        return 0.0
    else:
        return float(baseline)


# ===================== estimator (map generation) =====================


def run_estimator_new(
    arquivo_fits,
    error_map=None,  # error map
    *,
    nside=32,
    gal_chunk=50000,
    show_plot=False,
    output_dir="./results",
    eps=1e-12,
    sigma_col="sigma",  # name of the per-galaxy sigma column (radians)
    half_sky=True,      # True = northern hemisphere; False = full-sky
    nest=True           # HEALPix ordering scheme
):
    """
    Build HEALPix maps of the E-fields (E^a, E^b, E^{ab}, E^{a-b}) from a catalog.

    Observational case:
      - If the FITS table contains a column named `sigma`, use it as the
        per-galaxy angular uncertainty and derive weights from 1/sigma^2.
      - Otherwise, use `error_map` (if provided) to assign sigma per source.

    Parameters
    ----------
    arquivo_fits : str
        Path to the input FITS catalog containing the columns:
        g_rot_x, g_rot_y, g_rot_z, n_x, n_y, n_z, and optionally `sigma`.
    error_map : ndarray or None, optional
        HEALPix map of sigma values (radians), used if `sigma_col`
        is not present in the catalog.
    nside : int, optional
        NSIDE of the output HEALPix maps.
    gal_chunk : int, optional
        Number of galaxies processed per chunk (for memory control).
    show_plot : bool, optional
        If True, display Mollweide maps of E^a, E^b, E^{ab}, and E^{a-b}.
    output_dir : str, optional
        Directory where the output maps and stats file are written.
    eps : float, optional
        Small positive number used to regularize denominators.
    sigma_col : str, optional
        Name of the column containing per-galaxy sigma in radians.
    half_sky : bool, optional
        If True, only the northern hemisphere (z >= 0) is evaluated.
        If False, all-sky maps are computed.
    nest : bool, optional
        If True, use NESTED pixel ordering. The output FITS headers
        are tagged as "NESTED".

    Outputs
    -------
    The function writes the following FITS files to `output_dir`:

        <base>_Ea.fits
        <base>_Eb.fits
        <base>_Eab.fits
        <base>_Ea_minus_b.fits

    and a text file:

        <base>_stats.txt

    with C_BAR, SIGMA_MEAN, SIGMA_WMEAN, N_GAL, and HALFSKY.
    """
    t0 = time.time()

    # ---------- read catalog ----------
    with fits.open(arquivo_fits, memmap=True) as hdul:
        data = hdul[1].data
        a = np.column_stack(
            (
                np.asarray(data["g_rot_x"], dtype=np.float64),
                np.asarray(data["g_rot_y"], dtype=np.float64),
                np.asarray(data["g_rot_z"], dtype=np.float64),
            )
        )
        n = np.column_stack(
            (
                np.asarray(data["n_x"], dtype=np.float64),
                np.asarray(data["n_y"], dtype=np.float64),
                np.asarray(data["n_z"], dtype=np.float64),
            )
        )
        has_sigma = sigma_col in data.names
        if has_sigma:
            sigma_gal = np.asarray(data[sigma_col], dtype=np.float64)
        else:
            sigma_gal = None

    # Define b as the orthogonal tangential direction
    b = np.cross(n, a)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    b /= np.maximum(b_norm, 1e-30)

    # ---------- weights / C̄ ----------
    if sigma_gal is not None:
        sigma = _safe_sigma_floor(sigma_gal)
    else:
        if error_map is None:
            raise ValueError(
                "Neither a 'sigma' column in the FITS catalog nor an 'error_map' "
                "was provided."
            )
        nside_err = hp.npix2nside(error_map.size)
        error_map = np.asarray(error_map, dtype=np.float64)
        pix_err = hp.vec2pix(nside_err, n[:, 0], n[:, 1], n[:, 2], nest=nest)
        sigma = _safe_sigma_floor(error_map[pix_err])

    inv_var = 1.0 / (sigma**2)
    weights = inv_var / np.sum(inv_var)

    c = np.exp(-2.0 * (sigma**2))
    c_bar = float(np.sum(weights * c))
    sigma_mean = float(np.mean(sigma))
    sigma_wmean = float(np.sum(weights * sigma))

    # ---------- HEALPix grid ----------
    npix_total = hp.nside2npix(nside)
    ipix = np.arange(npix_total)
    x, y, z = hp.pix2vec(nside, ipix, nest=nest)
    d_all = np.vstack((x, y, z)).T.astype(np.float64, copy=False)
    mask = (d_all[:, 2] >= 0) if half_sky else np.ones(npix_total, dtype=bool)
    d_eval = d_all[mask]
    P = d_eval.shape[0]

    # ---------- accumulators ----------
    Ea = np.zeros(P, dtype=np.float64)
    Eb = np.zeros(P, dtype=np.float64)
    Eab = np.zeros(P, dtype=np.float64)

    Gtot = a.shape[0]
    nchunks = int(np.ceil(Gtot / gal_chunk))

    for k, g0 in enumerate(range(0, Gtot, gal_chunk), start=1):
        g1 = min(g0 + gal_chunk, Gtot)
        a_chunk = a[g0:g1]
        b_chunk = b[g0:g1]
        n_chunk = n[g0:g1]
        w_chunk = weights[g0:g1]

        # Project direction d onto the normal n
        dot_dn = d_eval @ n_chunk.T
        denom = 1.0 - dot_dn * dot_dn
        np.maximum(denom, eps, out=denom)
        np.sqrt(denom, out=denom)  # denom = ||d_t||

        del dot_dn

        dot_da_raw = d_eval @ a_chunk.T
        dot_db_raw = d_eval @ b_chunk.T

        # E^a
        tmp = (dot_da_raw / denom)
        tmp *= tmp
        tmp *= w_chunk
        Ea += tmp.sum(axis=1, dtype=np.float64)

        # E^b
        tmp = (dot_db_raw / denom)
        tmp *= tmp
        tmp *= w_chunk
        Eb += tmp.sum(axis=1, dtype=np.float64)

        # E^{ab} = 2 < (d·a)(d·b) / ||d_t||^2 >
        tmp = (dot_da_raw * dot_db_raw) / (denom * denom)
        tmp *= w_chunk
        Eab += 2.0 * tmp.sum(axis=1, dtype=np.float64)

        del a_chunk, b_chunk, n_chunk, w_chunk, denom, dot_da_raw, dot_db_raw, tmp
        print(
            f"[ESTIMATOR] Chunk {k}/{nchunks} "
            f"({100 * k / nchunks:.1f}%)  |  P={P}, G={g1 - g0}  "
            f"|  t={time.time() - t0:.1f}s"
        )

    # ---------- full-sky maps ----------
    Ea_full = np.zeros(npix_total, dtype=np.float64)
    Eb_full = np.zeros(npix_total, dtype=np.float64)
    Eab_full = np.zeros(npix_total, dtype=np.float64)
    Eaminusb_full = np.zeros(npix_total, dtype=np.float64)

    Ea_full[mask] = Ea
    Eb_full[mask] = Eb
    Eab_full[mask] = Eab
    Eaminusb_full[mask] = Ea - Eb

    if show_plot:
        hp.mollview(Ea_full, title="E^a map", unit="arb", nest=True)
        plt.show()
        hp.mollview(Eb_full, title="E^b map", unit="arb", nest=True)
        plt.show()
        hp.mollview(Eab_full, title="E^{ab} map", unit="arb", nest=True)
        plt.show()
        hp.mollview(Eaminusb_full, title="E^{a-b} map", unit="arb", nest=True)
        plt.show()

    # ---------- save output maps ----------
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(arquivo_fits))[0]

    def _write(name, arr):
        hdu = fits.PrimaryHDU(arr.astype(np.float64, copy=False))
        hdr = hdu.header
        hdr["NSIDE"] = nside
        hdr["ORDERING"] = "NESTED"
        hdr["SIMFILE"] = os.path.basename(arquivo_fits)
        hdr["DTYPE"] = "float64"
        out = os.path.join(output_dir, f"{base}_{name}.fits")
        hdu.writeto(out, overwrite=True)
        print(f"[WRITE] Saved map: {out}")

    _write("Ea", Ea_full)
    _write("Eb", Eb_full)
    _write("Eab", Eab_full)
    _write("Ea_minus_b", Eaminusb_full)

    # stats.txt with C̄ and sigma statistics
    stats_path = os.path.join(output_dir, f"{base}_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"C_BAR {c_bar:.16e}\n")
        f.write(f"SIGMA_MEAN {sigma_mean:.16e}\n")
        f.write(f"SIGMA_WMEAN {sigma_wmean:.16e}\n")
        f.write(f"N_GAL {Gtot}\n")
        f.write(f"HALFSKY {int(half_sky)}\n")
    print(f"[WRITE] Saved stats: {stats_path}")

    print(f"[ESTIMATOR] Total runtime: {time.time() - t0:.2f}s")


# ===================== analysis without PSF =====================


def analyze_dual_and_DIA(
    Ea,
    Eb,
    Eab,
    E_a_minus_b,
    *,
    nside,
    c_bar,
    nest=True,
    mask=None,
    eps=1e-15,
):
    """
    Analyze the dual E-field and intrinsic-alignment direction without PSF correction.

    Definitions
    -----------
    E(d) = E^{a-b}(d) + i E^{ab}(d)

    D_IA   = argmax_d |E(d)|
    A_IA   = |E(D_IA)| / C̄

    d_IA^a = argmax_d E^a(d)
    θ_IA^a = recover_alpha(E^a(d_IA^a), C̄)

    d_IA^b = argmax_d E^b(d)
    θ_IA^b = recover_alpha(E^b(d_IA^b), C̄)

    Parameters
    ----------
    Ea, Eb, Eab, E_a_minus_b : array-like
        HEALPix maps of the E-fields and their combinations.
    nside : int
        NSIDE of the maps.
    c_bar : float
        Effective concentration C̄.
    nest : bool, optional
        If True, use NESTED ordering.
    mask : array-like of bool or None, optional
        Optional mask specifying valid pixels; if None, a support
        mask is built from the input maps.
    eps : float, optional
        Small positive number used to regularize denominators.

    Returns
    -------
    result : dict
        Dictionary containing:
          - "D_IA": dict with best direction for |E|
          - "A_IA": amplitude at D_IA
          - "Eam_at_D", "Eam_over_W_at_D"
          - "d_IA_a", "theta_IA_a_arcmin", "Eam_at_da", "Eam_over_W_at_da"
          - "d_IA_b", "theta_IA_b_arcmin", "Eam_at_db", "Eam_over_W_at_db"
    """
    Ea = np.asarray(Ea, dtype=np.float64)
    Eb = np.asarray(Eb, dtype=np.float64)
    Eab = np.asarray(Eab, dtype=np.float64)
    Eam = np.asarray(E_a_minus_b, dtype=np.float64)

    if mask is None:
        mask = _support_mask(Ea, Eb, Eab, Eam)
    if not np.any(mask):
        raise ValueError("Mask has no valid pixels.")

    W = Ea + Eb
    E_complex = Eam + 1j * Eab
    modE = np.abs(E_complex)

    # D_IA (spin-2 axis)
    ipix_D = int(np.argmax(np.where(mask, modE, -np.inf)))
    A_IA = float(modE[ipix_D] / max(c_bar, eps))
    ra_D, dec_D, l_D, b_D = _pix_to_coords(nside, ipix_D, nest=nest)
    Eam_at_D = float(Eam[ipix_D])
    Eam_over_W_at_D = float(Eam[ipix_D] / max(W[ipix_D], eps))

    # d_IA^a
    ipix_da = int(np.argmax(np.where(mask, Ea, -np.inf)))
    m_a = float(Ea[ipix_da])
    theta_a_arcmin = float(np.degrees(recover_alpha(m_a, c_bar)) * 60.0)
    ra_a, dec_a, l_a, b_a = _pix_to_coords(nside, ipix_da, nest=nest)
    Eam_at_da = float(Eam[ipix_da])
    Eam_over_W_at_da = float(Eam[ipix_da] / max(W[ipix_da], eps))

    # d_IA^b
    ipix_db = int(np.argmax(np.where(mask, Eb, -np.inf)))
    m_b = float(Eb[ipix_db])
    theta_b_arcmin = float(np.degrees(recover_alpha(m_b, c_bar)) * 60.0)
    ra_b, dec_b, l_b, b_b = _pix_to_coords(nside, ipix_db, nest=nest)
    Eam_at_db = float(Eam[ipix_db])
    Eam_over_W_at_db = float(Eam[ipix_db] / max(W[ipix_db], eps))

    return {
        # D_IA and amplitude
        "D_IA": {
            "pix": ipix_D,
            "ra_deg": ra_D,
            "dec_deg": dec_D,
            "l_deg": l_D,
            "b_deg": b_D,
        },
        "A_IA": A_IA,
        "Eam_at_D": Eam_at_D,
        "Eam_over_W_at_D": Eam_over_W_at_D,
        # a-axis
        "d_IA_a": {
            "pix": ipix_da,
            "ra_deg": ra_a,
            "dec_deg": dec_a,
            "l_deg": l_a,
            "b_deg": b_a,
        },
        "theta_IA_a_arcmin": theta_a_arcmin,
        "Eam_at_da": Eam_at_da,
        "Eam_over_W_at_da": Eam_over_W_at_da,
        # b-axis
        "d_IA_b": {
            "pix": ipix_db,
            "ra_deg": ra_b,
            "dec_deg": dec_b,
            "l_deg": l_b,
            "b_deg": b_b,
        },
        "theta_IA_b_arcmin": theta_b_arcmin,
        "Eam_at_db": Eam_at_db,
        "Eam_over_W_at_db": Eam_over_W_at_db,
    }


# ===================== with PSF =====================

def fit_eta_complex(Eam_obs, Eab_obs, Eam_psf, Eab_psf, *, mask=None, eps=1e-15):
    """
    Fit complex eta (spin-2 leakage) via linear regression on E^{a-b} + i E^{ab}.

    Parameters
    ----------
    Eam_obs, Eab_obs : array-like
        Observed E^{a-b} and E^{ab} maps.
    Eam_psf, Eab_psf : array-like
        PSF E^{a-b} and E^{ab} maps.
    mask : array-like of bool or None, optional
        Optional mask for valid pixels. If None, built from support.
    eps : float, optional
        Regularization parameter for small denominators.

    Returns
    -------
    eta_complex : complex
        Complex leakage parameter η (spin-2).
    """
    if mask is None:
        mask = _support_mask(Eam_obs, Eab_obs) & _support_mask(Eam_psf, Eab_psf)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0.0 + 0.0j

    y = (np.asarray(Eam_obs) + 1j * np.asarray(Eab_obs))[idx]
    x = (np.asarray(Eam_psf) + 1j * np.asarray(Eab_psf))[idx]

    # Centered regression (remove mean from x and y)
    mx = x.mean()
    my = y.mean()
    x0 = x - mx
    y0 = y - my
    den = np.vdot(x0, x0).real
    return 0.0 + 0.0j if den < eps else np.vdot(x0, y0) / den


def fit_eta_real_axiswise(E_obs, E_psf, *, mask=None, eps=1e-15):
    """
    Fit real eta (per axis) via linear regression on a single E-field component.

    Parameters
    ----------
    E_obs : array-like
        Observed E-field (E^a or E^b).
    E_psf : array-like
        PSF E-field (same component).
    mask : array-like of bool or None, optional
        Optional mask for valid pixels. If None, built from support.
    eps : float, optional
        Regularization parameter for small denominators.

    Returns
    -------
    eta_real : float
        Real leakage parameter for the given axis.
    """
    if mask is None:
        mask = _support_mask(E_obs) & _support_mask(E_psf)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0.0

    x = np.asarray(E_psf)[idx]
    y = np.asarray(E_obs)[idx]

    # Centered regression
    mx = x.mean()
    my = y.mean()
    x0 = x - mx
    y0 = y - my
    den = np.sum(x0 * x0)
    return 0.0 if den < eps else float(np.dot(x0, y0) / den)


def analyze_with_psf_corrections(
    Ea_obs,
    Eb_obs,
    Eab_obs,
    Ea_psf,
    Eb_psf,
    Eab_psf,
    *,
    nside,
    c_bar,
    nest=True,
    eps=1e-15,
    de_leak_mode="baseline",
    baseline_real=0.5,
    baseline_complex=0.0,
):
    """
    Perform PSF-leakage corrections on the E-fields using complex and real eta.

    Strategy
    --------
    (A) Spin-2 (complex η):
        - Fit η_c from (E^{a-b} + i E^{ab})_obs vs PSF.
        - De-leak using either a "centered" or "baseline" approach.

    (B) Axis-wise (η_a, η_b):
        - Fit η_a from E^a_obs vs E^a_psf.
        - Fit η_b from E^b_obs vs E^b_psf.
        - De-leak using the same `de_leak_mode` logic for each axis.

    Parameters
    ----------
    Ea_obs, Eb_obs, Eab_obs : array-like
        Observed E^a, E^b and E^{ab} maps.
    Ea_psf, Eb_psf, Eab_psf : array-like
        PSF E^a, E^b and E^{ab} maps.
    nside : int
        NSIDE of the maps.
    c_bar : float
        Effective concentration C̄.
    nest : bool, optional
        If True, use NESTED ordering.
    eps : float, optional
        Regularization parameter for small denominators.
    de_leak_mode : {"baseline", "centered"}, optional
        - "centered": subtract means from both obs and PSF before fitting.
        - "baseline": subtract an explicit baseline (real or complex).
    baseline_real : float or str, optional
        Real baseline used for E^a and E^b. If "mean", uses the masked
        mean of the PSF map. If numeric, used directly.
    baseline_complex : float or str, optional
        Complex baseline (implemented as a scalar) for the complex spin-2
        de-leaking step. If "mean", uses the masked mean of x_c.

    Returns
    -------
    result : dict
        Dictionary containing:
          - "eta_complex"
          - "D_IA", "A_IA", "Eam_at_D", "Eam_over_W_at_D"
          - "eta_a", "eta_b"
          - "d_IA_a", "theta_IA_a_arcmin", "Eam_at_da", "Eam_over_W_at_da"
          - "d_IA_b", "theta_IA_b_arcmin", "Eam_at_db", "Eam_over_W_at_db"
          - Residual maps: Ea_res_c, Eb_res_c, Eam_res, Eab_res,
            Ea_res_ax, Eb_res_ax, Eam_res_ax
    """
    Ea_obs = np.asarray(Ea_obs, dtype=np.float64)
    Eb_obs = np.asarray(Eb_obs, dtype=np.float64)
    Eab_obs = np.asarray(Eab_obs, dtype=np.float64)
    Ea_psf = np.asarray(Ea_psf, dtype=np.float64)
    Eb_psf = np.asarray(Eb_psf, dtype=np.float64)
    Eab_psf = np.asarray(Eab_psf, dtype=np.float64)

    mask_obs = _support_mask(Ea_obs, Eb_obs, Eab_obs)
    mask_psf = _support_mask(Ea_psf, Eb_psf, Eab_psf)
    mask_all = mask_obs & mask_psf
    if not np.any(mask_all):
        raise ValueError("Intersection OBS ∩ PSF is empty in the northern hemisphere.")

    # (A) Spin-2 (complex η): fit η with centering
    Eam_obs, Eam_psf = Ea_obs - Eb_obs, Ea_psf - Eb_psf
    y_c = Eam_obs + 1j * Eab_obs
    x_c = Eam_psf + 1j * Eab_psf
    mx = x_c[mask_all].mean()
    my = y_c[mask_all].mean()
    eta_c = fit_eta_complex(Eam_obs, Eab_obs, Eam_psf, Eab_psf, mask=mask_all, eps=eps)

    # --- de-leak: baseline or centered ---
    if de_leak_mode.lower() == "centered":
        # Remove global means explicitly
        E_res_c = (y_c - my) - eta_c * (x_c - mx)
    else:
        # "baseline" mode: use explicit baseline for complex de-leak (default 0.0)
        b_c = _resolve_baseline(baseline_complex, x_c, mask_all)
        E_res_c = y_c - eta_c * (x_c - b_c)

    Eam_res = np.real(E_res_c)
    Eab_res = np.imag(E_res_c)
    modE = np.abs(E_res_c)

    ipix_D = int(np.argmax(np.where(mask_all, modE, -np.inf)))
    A_IA = float(modE[ipix_D] / max(c_bar, eps))
    ra_D, dec_D, l_D, b_D = _pix_to_coords(nside, ipix_D, nest=nest)
    Wc = Ea_obs + Eb_obs
    Eam_at_D = float(Eam_res[ipix_D])
    Eam_over_W_at_D = float(Eam_res[ipix_D] / max(Wc[ipix_D], eps))

    # Residual maps consistent with spin-2 decomposition
    Ea_res_c = 0.5 * (Wc + Eam_res)
    Eb_res_c = 0.5 * (Wc - Eam_res)

    # (B) Axis-wise (η_a, η_b) with centered fit and real baseline
    mask_a = _support_mask(Ea_obs) & _support_mask(Ea_psf) & mask_obs
    mask_b = _support_mask(Eb_obs) & _support_mask(Eb_psf) & mask_obs
    eta_a = fit_eta_real_axiswise(Ea_obs, Ea_psf, mask=mask_a, eps=eps)
    eta_b = fit_eta_real_axiswise(Eb_obs, Eb_psf, mask=mask_b, eps=eps)

    if de_leak_mode.lower() == "centered":
        mx_a = Ea_psf[mask_a].mean()
        my_a = Ea_obs[mask_a].mean()
        Ea_res_ax = (Ea_obs - my_a) - eta_a * (Ea_psf - mx_a)

        mx_b = Eb_psf[mask_b].mean()
        my_b = Eb_obs[mask_b].mean()
        Eb_res_ax = (Eb_obs - my_b) - eta_b * (Eb_psf - mx_b)
    else:
        # "baseline" mode: typical real baseline is 0.5
        b_a = _resolve_baseline(baseline_real, Ea_psf, mask_a)
        b_b = _resolve_baseline(baseline_real, Eb_psf, mask_b)
        Ea_res_ax = Ea_obs - eta_a * (Ea_psf - b_a)
        Eb_res_ax = Eb_obs - eta_b * (Eb_psf - b_b)

    Eam_res_ax = Ea_res_ax - Eb_res_ax
    Wax = Ea_res_ax + Eb_res_ax

    ipix_da = int(np.argmax(np.where(mask_a, Ea_res_ax, -np.inf)))
    ra_a, dec_a, l_a, b_a_gal = _pix_to_coords(nside, ipix_da, nest=nest)
    theta_a_arcmin = float(
        np.degrees(recover_alpha(float(Ea_res_ax[ipix_da]), c_bar)) * 60.0
    )
    Eam_at_da = float(Eam_res_ax[ipix_da])
    Eam_over_W_at_da = float(Eam_res_ax[ipix_da] / max(Wax[ipix_da], eps))

    ipix_db = int(np.argmax(np.where(mask_b, Eb_res_ax, -np.inf)))
    ra_b, dec_b, l_b, b_b_gal = _pix_to_coords(nside, ipix_db, nest=nest)
    theta_b_arcmin = float(
        np.degrees(recover_alpha(float(Eb_res_ax[ipix_db]), c_bar)) * 60.0
    )
    Eam_at_db = float(Eam_res_ax[ipix_db])
    Eam_over_W_at_db = float(Eam_res_ax[ipix_db] / max(Wax[ipix_db], eps))

    return {
        # Spin-2
        "eta_complex": complex(eta_c),
        "D_IA": {
            "pix": ipix_D,
            "ra_deg": ra_D,
            "dec_deg": dec_D,
            "l_deg": l_D,
            "b_deg": b_D,
        },
        "A_IA": A_IA,
        "Eam_at_D": Eam_at_D,
        "Eam_over_W_at_D": Eam_over_W_at_D,
        # Axis-wise (real)
        "eta_a": float(eta_a),
        "eta_b": float(eta_b),
        "d_IA_a": {
            "pix": ipix_da,
            "ra_deg": ra_a,
            "dec_deg": dec_a,
            "l_deg": l_a,
            "b_deg": b_a_gal,
        },
        "theta_IA_a_arcmin": theta_a_arcmin,
        "Eam_at_da": Eam_at_da,
        "Eam_over_W_at_da": Eam_over_W_at_da,
        "d_IA_b": {
            "pix": ipix_db,
            "ra_deg": ra_b,
            "dec_deg": dec_b,
            "l_deg": l_b,
            "b_deg": b_b_gal,
        },
        "theta_IA_b_arcmin": theta_b_arcmin,
        "Eam_at_db": Eam_at_db,
        "Eam_over_W_at_db": Eam_over_W_at_db,
        # Residual maps (for diagnostics)
        "Ea_res_c": Ea_res_c,
        "Eb_res_c": Eb_res_c,
        "Eam_res": Eam_res,
        "Eab_res": Eab_res,
        "Ea_res_ax": Ea_res_ax,
        "Eb_res_ax": Eb_res_ax,
        "Eam_res_ax": Eam_res_ax,
    }


# ===================== per-simulation FITS output =====================


def _save_result_fits_per_sim(
    base_path_no_suffix,
    row_dict,
    *,
    nside,
    nest,
    cbar,
    stats,
    suffix,
):
    """
    Save a one-row FITS table with analysis results for a single simulation/case.

    Parameters
    ----------
    base_path_no_suffix : str
        Base path (without suffix) for the output file.
    row_dict : dict
        Dictionary of columns to write as a single-row table.
    nside : int
        NSIDE of the maps used in the analysis.
    nest : bool
        Whether maps are in NESTED (True) or RING (False) ordering.
    cbar : float
        Effective concentration C̄.
    stats : dict
        Dictionary of scalar statistics (SIGMA_MEAN, SIGMA_WMEAN, N_GAL, etc.).
    suffix : str
        Suffix to append to the filename (e.g. "nopsf", "psf", "obs_psf").
    """
    t = Table(rows=[row_dict])
    out = base_path_no_suffix + f"_analysis_{suffix}.fits"
    t.write(out, format="fits", overwrite=True)
    with fits.open(out, mode="update") as hdul:
        hdr = hdul[1].header
        hdr["NSIDE"] = int(nside)
        hdr["ORDERING"] = "NESTED" if nest else "RING"
        hdr["C_BAR"] = float(cbar)
        if "SIGMA_MEAN" in stats:
            hdr["S_MEAN"] = float(stats["SIGMA_MEAN"])
        if "SIGMA_WMEAN" in stats:
            hdr["S_WMEAN"] = float(stats["SIGMA_WMEAN"])
        if "N_GAL" in stats:
            hdr["N_GAL"] = int(stats["N_GAL"])
        hdul.flush()
    print(f"[FITS] Saved per-simulation analysis file: {out}")


# ===================== batch: without PSF =====================


def batch_analyze(
    output_dir,
    *,
    prefix="simulation_d_xyz_t_0_sim_",
    sims=range(10),
    nside_default=32,
    nest=True,
    save_per_sim_fits=True,
    save_summary_fits=True,
):
    """
    Batch analysis (without PSF corrections) over a set of simulations.

    Parameters
    ----------
    output_dir : str
        Directory where the E-maps and stats.txt files are stored.
    prefix : str, optional
        Filename prefix for simulations; maps are expected as
        <output_dir>/<prefix><sim_index>_Ea.fits, etc.
    sims : iterable, optional
        Iterable of simulation indices.
    nside_default : int, optional
        Default NSIDE if not found in the FITS header.
    nest : bool, optional
        Whether to assume NESTED ordering.
    save_per_sim_fits : bool, optional
        If True, save one FITS table per simulation with the analysis results.
    save_summary_fits : bool, optional
        If True, save a summary FITS table across all simulations.

    Returns
    -------
    summary_table : astropy.table.Table
        In-memory summary table with one row per simulation.
    """
    rows = []
    for j in sims:
        base = os.path.join(output_dir, f"{prefix}{j}")
        p_Ea = base + "_Ea.fits"
        p_Eb = base + "_Eb.fits"
        p_Eab = base + "_Eab.fits"
        p_Eam = base + "_Ea_minus_b.fits"  # if missing, computed as Ea - Eb
        p_stat = base + "_stats.txt"

        Cbar, stats = _read_cbar(p_stat)
        Ea = fits.getdata(p_Ea)
        Eb = fits.getdata(p_Eb)
        Eab = fits.getdata(p_Eab)
        if os.path.exists(p_Eam):
            Eam = fits.getdata(p_Eam)
        else:
            Eam = Ea - Eb

        try:
            hdr = fits.getheader(p_Ea)
            nside = int(hdr.get("NSIDE", nside_default))
        except Exception:
            nside = nside_default

        res = analyze_dual_and_DIA(Ea, Eb, Eab, Eam, nside=nside, c_bar=Cbar, nest=nest)

        row = {
            "sim": j,
            "nside": nside,
            "C_bar": Cbar,
            # D_IA
            "D_pix": res["D_IA"]["pix"],
            "D_ra_deg": res["D_IA"]["ra_deg"],
            "D_dec_deg": res["D_IA"]["dec_deg"],
            "D_l_deg": res["D_IA"]["l_deg"],
            "D_b_deg": res["D_IA"]["b_deg"],
            "A_IA": res["A_IA"],
            "Eam_at_D": res["Eam_at_D"],
            "Eam_over_W_at_D": res["Eam_over_W_at_D"],
            # d_IA^a
            "da_pix": res["d_IA_a"]["pix"],
            "da_ra_deg": res["d_IA_a"]["ra_deg"],
            "da_dec_deg": res["d_IA_a"]["dec_deg"],
            "da_l_deg": res["d_IA_a"]["l_deg"],
            "da_b_deg": res["d_IA_a"]["b_deg"],
            "theta_a_arcmin": res["theta_IA_a_arcmin"],
            "Eam_at_da": res["Eam_at_da"],
            "Eam_over_W_at_da": res["Eam_over_W_at_da"],
            # d_IA^b
            "db_pix": res["d_IA_b"]["pix"],
            "db_ra_deg": res["d_IA_b"]["ra_deg"],
            "db_dec_deg": res["d_IA_b"]["dec_deg"],
            "db_l_deg": res["d_IA_b"]["l_deg"],
            "db_b_deg": res["d_IA_b"]["b_deg"],
            "theta_b_arcmin": res["theta_IA_b_arcmin"],
            "Eam_at_db": res["Eam_at_db"],
            "Eam_over_W_at_db": res["Eam_over_W_at_db"],
        }
        rows.append(row)

        if save_per_sim_fits:
            _save_result_fits_per_sim(
                base, row, nside=nside, nest=nest, cbar=Cbar, stats=stats, suffix="nopsf"
            )

        print(f"[BATCH nopsf] sim {j} | A_IA={row['A_IA']:.4f}")

    tab = Table(rows=rows)
    out_ecsv = os.path.join(output_dir, "summary_IA_nopsf.ascii.ecsv")
    tab.write(out_ecsv, format="ascii.ecsv", overwrite=True)
    print(f"[SUMMARY nopsf] ECSV summary saved: {out_ecsv}")
    if save_summary_fits:
        out_fits = os.path.join(output_dir, "summary_IA_nopsf.fits")
        tab.write(out_fits, format="fits", overwrite=True)
        print(f"[SUMMARY nopsf] FITS summary saved: {out_fits}")
    return tab


# ===================== batch: with PSF =====================


def batch_analyze_with_psf(
    output_dir,
    *,
    prefix="simulation_d_xyz_t_0_sim_",
    sims=range(10),
    psf_suffix="_psf",
    nside_default=32,
    nest=True,
    save_per_sim_fits=True,
    save_summary_fits=True,
):
    """
    Batch analysis with PSF corrections over a set of simulations.

    Parameters
    ----------
    output_dir : str
        Directory where the observed and PSF E-maps and stats.txt files are stored.
    prefix : str, optional
        Filename prefix for the observed simulations.
    sims : iterable, optional
        Iterable of simulation indices.
    psf_suffix : str, optional
        Suffix appended to the base name for PSF maps.
    nside_default : int, optional
        Default NSIDE if not found in headers.
    nest : bool, optional
        Whether to assume NESTED ordering.
    save_per_sim_fits : bool, optional
        If True, save one FITS table per simulation (with PSF corrections).
    save_summary_fits : bool, optional
        If True, save a FITS summary across all simulations.

    Returns
    -------
    summary_table : astropy.table.Table
        In-memory summary table with one row per simulation.
    """
    rows = []
    for j in sims:
        base = os.path.join(output_dir, f"{prefix}{j}")
        base_psf = base + psf_suffix

        p_Ea = base + "_Ea.fits"
        p_Eb = base + "_Eb.fits"
        p_Eab = base + "_Eab.fits"
        p_stat = base + "_stats.txt"

        p_Ea_psf = base_psf + "_Ea.fits"
        p_Eb_psf = base_psf + "_Eb.fits"
        p_Eab_psf = base_psf + "_Eab.fits"

        Cbar, stats = _read_cbar(p_stat)

        Ea_obs = fits.getdata(p_Ea)
        Eb_obs = fits.getdata(p_Eb)
        Eab_obs = fits.getdata(p_Eab)
        Ea_psf = fits.getdata(p_Ea_psf)
        Eb_psf = fits.getdata(p_Eb_psf)
        Eab_psf = fits.getdata(p_Eab_psf)

        try:
            hdr = fits.getheader(p_Ea)
            nside = int(hdr.get("NSIDE", nside_default))
        except Exception:
            nside = nside_default

        res = analyze_with_psf_corrections(
            Ea_obs,
            Eb_obs,
            Eab_obs,
            Ea_psf,
            Eb_psf,
            Eab_psf,
            nside=nside,
            c_bar=Cbar,
            nest=nest,
        )

        row = {
            "sim": j,
            "nside": nside,
            "C_bar": Cbar,
            # spin-2 (complex eta)
            "eta_c_real": float(np.real(res["eta_complex"])),
            "eta_c_imag": float(np.imag(res["eta_complex"])),
            "eta_c_abs": float(np.abs(res["eta_complex"])),
            "eta_c_arg_deg": float(
                np.degrees(np.angle(res["eta_complex"]))
            ) if np.abs(res["eta_complex"]) > 0 else 0.0,
            "D_pix": res["D_IA"]["pix"],
            "D_ra_deg": res["D_IA"]["ra_deg"],
            "D_dec_deg": res["D_IA"]["dec_deg"],
            "D_l_deg": res["D_IA"]["l_deg"],
            "D_b_deg": res["D_IA"]["b_deg"],
            "A_IA": res["A_IA"],
            "Eam_at_D": res["Eam_at_D"],
            "Eam_over_W_at_D": res["Eam_over_W_at_D"],
            # axis-wise
            "eta_a": res["eta_a"],
            "eta_b": res["eta_b"],
            "da_pix": res["d_IA_a"]["pix"],
            "da_ra_deg": res["d_IA_a"]["ra_deg"],
            "da_dec_deg": res["d_IA_a"]["dec_deg"],
            "da_l_deg": res["d_IA_a"]["l_deg"],
            "da_b_deg": res["d_IA_a"]["b_deg"],
            "theta_a_arcmin": res["theta_IA_a_arcmin"],
            "Eam_at_da": res["Eam_at_da"],
            "Eam_over_W_at_da": res["Eam_over_W_at_da"],
            "db_pix": res["d_IA_b"]["pix"],
            "db_ra_deg": res["d_IA_b"]["ra_deg"],
            "db_dec_deg": res["d_IA_b"]["dec_deg"],
            "db_l_deg": res["d_IA_b"]["l_deg"],
            "db_b_deg": res["d_IA_b"]["b_deg"],
            "theta_b_arcmin": res["theta_IA_b_arcmin"],
            "Eam_at_db": res["Eam_at_db"],
            "Eam_over_W_at_db": res["Eam_over_W_at_db"],
        }
        rows.append(row)

        if save_per_sim_fits:
            _save_result_fits_per_sim(
                base, row, nside=nside, nest=nest, cbar=Cbar, stats=stats, suffix="psf"
            )

        print(
            f"[BATCH psf] sim {j} | eta_c=({row['eta_c_real']:.3g},"
            f"{row['eta_c_imag']:.3g}i)  A_IA={row['A_IA']:.4f}"
        )

    tab = Table(rows=rows)
    out_ecsv = os.path.join(output_dir, "summary_IA_with_psf.ascii.ecsv")
    tab.write(out_ecsv, format="ascii.ecsv", overwrite=True)
    print(f"[SUMMARY psf] ECSV summary saved: {out_ecsv}")
    if save_summary_fits:
        out_fits = os.path.join(output_dir, "summary_IA_with_psf.fits")
        tab.write(out_fits, format="fits", overwrite=True)
        print(f"[SUMMARY psf] FITS summary saved: {out_fits}")
    return tab


# ===================== bootstrap =====================


def _build_blocks_from_catalog(n_xyz, *, nside_blocks=16, nest=True):
    """
    Map each galaxy to a HEALPix pixel at NSIDE = nside_blocks (block bootstrap).

    Parameters
    ----------
    n_xyz : ndarray, shape (N, 3)
        Unit vectors (x, y, z) for each galaxy.
    nside_blocks : int, optional
        NSIDE used to define bootstrap blocks.
    nest : bool, optional
        If True, use NESTED pixel indexing.

    Returns
    -------
    block_ids : ndarray of int
        Sorted array of HEALPix pixel indices defining blocks.
    lists : list of ndarray
        For each block, an array of indices of galaxies belonging to it.
    """
    ipix = hp.vec2pix(nside_blocks, n_xyz[:, 0], n_xyz[:, 1], n_xyz[:, 2], nest=nest)
    blocks = {}
    for i, p in enumerate(ipix):
        blocks.setdefault(int(p), []).append(i)
    block_ids = np.array(sorted(blocks.keys()), dtype=int)
    lists = [np.array(blocks[i], dtype=int) for i in block_ids]
    return block_ids, lists


def _compute_maps_from_arrays(
    a,
    n,
    *,
    sigma_gal=None,
    error_map=None,  # per-galaxy sigma or map
    nside=32,
    half_sky=True,
    nest=True,
    gal_chunk=50000,
    eps=1e-12,
):
    """
    Minimal estimator for bootstrap, taking vectors (a, n) as input.

    Parameters
    ----------
    a : ndarray, shape (N, 3)
        Tangential unit vectors (galaxy orientation).
    n : ndarray, shape (N, 3)
        Line-of-sight unit vectors.
    sigma_gal : ndarray or None, optional
        Per-galaxy sigma (radians). If provided, takes precedence.
    error_map : ndarray or None, optional
        HEALPix map of sigma values (radians).
    nside : int, optional
        NSIDE of the output maps.
    half_sky : bool, optional
        If True, restrict evaluation to the northern hemisphere (z >= 0).
    nest : bool, optional
        If True, use NESTED ordering.
    gal_chunk : int, optional
        Number of galaxies per chunk.
    eps : float, optional
        Regularization parameter for denominators.

    Returns
    -------
    result : dict
        Dictionary with Ea, Eb, Eab, Eam maps and c_bar, nside.
    """
    b = np.cross(n, a)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-30)

    # weights
    if sigma_gal is not None:
        sigma = _safe_sigma_floor(sigma_gal)
    else:
        if error_map is None:
            raise ValueError(
                "Per-galaxy sigma is missing and 'error_map' was not provided."
            )
        nside_err = hp.npix2nside(error_map.size)
        pix_err = hp.vec2pix(nside_err, n[:, 0], n[:, 1], n[:, 2], nest=nest)
        sigma = _safe_sigma_floor(np.asarray(error_map, float)[pix_err])

    inv_var = 1.0 / (sigma**2)
    weights = inv_var / np.sum(inv_var)
    c_bar = float(np.sum(weights * np.exp(-2.0 * sigma**2)))

    # grid
    npix_total = hp.nside2npix(nside)
    x, y, z = hp.pix2vec(nside, np.arange(npix_total), nest=nest)
    d_all = np.vstack((x, y, z)).T
    mask = (d_all[:, 2] >= 0) if half_sky else np.ones(npix_total, dtype=bool)
    d = d_all[mask]
    P = d.shape[0]

    Ea = np.zeros(P)
    Eb = np.zeros(P)
    Eab = np.zeros(P)
    G = a.shape[0]
    for g0 in range(0, G, gal_chunk):
        g1 = min(g0 + gal_chunk, G)
        a_ch = a[g0:g1]
        b_ch = b[g0:g1]
        n_ch = n[g0:g1]
        w_ch = weights[g0:g1]

        dot_dn = d @ n_ch.T
        denom = np.sqrt(np.maximum(eps, 1.0 - dot_dn**2))  # ||d_t||
        dot_da = d @ a_ch.T
        dot_db = d @ b_ch.T
        Ea += ((dot_da / denom) ** 2 * w_ch).sum(axis=1)
        Eb += ((dot_db / denom) ** 2 * w_ch).sum(axis=1)
        Eab += (2.0 * ((dot_da * dot_db) / (denom * denom)) * w_ch).sum(axis=1)

    Ea_full = np.zeros(npix_total)
    Eb_full = np.zeros(npix_total)
    Eab_full = np.zeros(npix_total)
    Ea_full[mask] = Ea
    Eb_full[mask] = Eb
    Eab_full[mask] = Eab
    Eam_full = np.zeros(npix_total)
    Eam_full[mask] = Ea - Eb
    return dict(Ea=Ea_full, Eb=Eb_full, Eab=Eab_full, Eam=Eam_full, c_bar=c_bar, nside=nside)


# ---------- small axial rotation (for lensing injection) ----------


def _rotate_tangent_toward_dir(a, n, d_hat, theta_rad, eps=1e-12):
    """
    Rotate each tangential vector a_i (perpendicular to n_i) by an angle <= theta_rad
    towards the projection of d_hat onto the tangent plane at n_i.

    This is used to inject a small additional IA/lensing-like signal.

    Parameters
    ----------
    a : ndarray, shape (N, 3)
        Tangential unit vectors.
    n : ndarray, shape (N, 3)
        Line-of-sight unit vectors.
    d_hat : array-like, shape (3,)
        Target direction unit vector.
    theta_rad : float
        Maximum rotation angle in radians.
    eps : float, optional
        Small number for numerical stability.

    Returns
    -------
    a_rot : ndarray, shape (N, 3)
        Rotated and renormalized tangential vectors.
    """
    a = np.asarray(a, float, order="C")
    n = np.asarray(n, float, order="C")
    N = a.shape[0]
    d = np.broadcast_to(np.asarray(d_hat, float), (N, 3))
    dot_dn = np.einsum("ij,ij->i", n, d)
    t_target = d - dot_dn[:, None] * n
    t_norm = np.linalg.norm(t_target, axis=1)
    ok = t_norm > eps
    t_target[ok] /= t_norm[ok, None]
    t_target[~ok] = a[~ok]

    dot_gt = np.einsum("ij,ij->i", a, t_target)
    t_target[dot_gt < 0] *= -1
    dot_gt = np.clip(np.einsum("ij,ij->i", a, t_target), -1.0, 1.0)
    ang = np.arccos(dot_gt)
    cross_gt = np.cross(a, t_target)
    sign = np.sign(np.einsum("ij,ij->i", n, cross_gt))
    sign[sign == 0] = 1.0
    delta = np.minimum(ang, theta_rad)
    phi = sign * delta
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    n_cross_a = np.cross(n, a)
    a_rot = a * cos_phi[:, None] + n_cross_a * sin_phi[:, None]
    sat = ang < theta_rad
    a_rot[sat] = t_target[sat]
    a_rot /= np.maximum(np.linalg.norm(a_rot, axis=1, keepdims=True), 1e-30)
    return a_rot


def bootstrap_errors_from_catalog(
    arquivo_fits,
    error_map=None,  # optional error map
    *,
    arquivo_fits_psf=None,
    with_psf=False,
    lensing=False,
    lensing_theta_arcmin=1.0,
    lensing_nside_dir=4096,
    B=512,
    nside_blocks=16,
    nside_eval=32,
    nest=True,
    half_sky=True,  # True = north, False = full-sky
    outdir="results/boot",
    seed=2025,
    sigma_col="sigma",  # name of sigma column
    save_residual_maps=False,
    residual_dir=None,
):
    """
    Block bootstrap (HEALPix NSIDE=nside_blocks) from an input catalog.

    Behavior
    --------
    - If the catalog has a column `sigma` (radians), it is used as per-galaxy sigma.
    - Otherwise, `error_map` is used to assign sigma per source.
    - with_psf=True: requires `arquivo_fits_psf` with PSF g_rot_*.
    - lensing=True: applies an additional IA-like rotation with angle `lensing_theta_arcmin`
      around a random direction per realization.
    - half_sky: True for northern hemisphere; False for full-sky.
    - save_residual_maps: if True, save maps per realization for diagnostics.

    Parameters
    ----------
    arquivo_fits : str
        Path to the observed catalog FITS file.
    error_map : ndarray or None, optional
        HEALPix map of sigma values in radians, used if sigma_col is absent.
    arquivo_fits_psf : str or None, optional
        Path to the PSF catalog FITS file (matching g_rot_* structure).
    with_psf : bool, optional
        If True, run PSF corrections inside each bootstrap realization.
    lensing : bool, optional
        If True, inject additional IA-like rotation of amplitude lensing_theta_arcmin.
    lensing_theta_arcmin : float, optional
        Rotation angle in arcminutes for IA injection.
    lensing_nside_dir : int, optional
        NSIDE used to pick random directions for lensing injection.
    B : int, optional
        Number of bootstrap realizations.
    nside_blocks : int, optional
        NSIDE used to define blocks in the input catalog.
    nside_eval : int, optional
        NSIDE of the E-maps computed for each realization.
    nest : bool, optional
        Whether to use NESTED ordering for HEALPix.
    half_sky : bool, optional
        True to restrict to the northern hemisphere; False for full-sky.
    outdir : str, optional
        Output directory for bootstrap summary FITS files.
    seed : int, optional
        Seed for the random number generator.
    sigma_col : str, optional
        Column name containing per-galaxy sigma in radians.
    save_residual_maps : bool, optional
        If True, save residual E-maps per bootstrap realization.
    residual_dir : str or None, optional
        Directory where residual maps are saved. If None and
        save_residual_maps is True, a "resmaps" subdirectory is created.

    Returns
    -------
    out_fits : str
        Path to the output bootstrap summary FITS file.
    """
    os.makedirs(outdir, exist_ok=True)

    t0_boot = time.time()
    print(
        f"[BOOT] start  B={B}  nside_blocks={nside_blocks}  nside_eval={nside_eval}  "
        f"with_psf={with_psf}  lensing={lensing}  half_sky={half_sky}"
    )

    with fits.open(arquivo_fits, memmap=True) as hdul:
        d = hdul[1].data
        aO = np.column_stack((d["g_rot_x"], d["g_rot_y"], d["g_rot_z"])).astype(float)
        n = np.column_stack((d["n_x"], d["n_y"], d["n_z"])).astype(float)
        has_sigma = sigma_col in d.names
        sigma_gal = np.asarray(d[sigma_col], float) if has_sigma else None

    if with_psf:
        if arquivo_fits_psf is None:
            raise ValueError("arquivo_fits_psf is required when with_psf=True.")
        with fits.open(arquivo_fits_psf, memmap=True) as hdulp:
            dp = hdulp[1].data
            aP = np.column_stack((dp["g_rot_x"], dp["g_rot_y"], dp["g_rot_z"])).astype(
                float
            )

    block_ids, idx_lists = _build_blocks_from_catalog(
        n, nside_blocks=nside_blocks, nest=nest
    )
    Nb = len(idx_lists)
    rng = np.random.default_rng(seed)

    # helper to save maps per iteration
    def _save_map_fits(path, arr, *, nside, order="NESTED"):
        hdu = fits.PrimaryHDU(np.asarray(arr, dtype=np.float64))
        hdr = hdu.header
        hdr["NSIDE"] = int(nside)
        hdr["ORDERING"] = str(order)
        hdu.writeto(path, overwrite=True)

    if save_residual_maps:
        if residual_dir is None:
            residual_dir = os.path.join(outdir, "resmaps")
        os.makedirs(residual_dir, exist_ok=True)
        print(f"[BOOT] saving residual maps per iteration in: {residual_dir}")

    rows = []
    base = os.path.splitext(os.path.basename(arquivo_fits))[0]
    mode = ("withpsf" if with_psf else "nopsf") + ("_lens" if lensing else "")
    theta_L = np.deg2rad(lensing_theta_arcmin / 60.0)
    npix_dir = hp.nside2npix(lensing_nside_dir)

    for b in range(B):
        t1_iter = time.time()
        print(f"[BOOT {b+1}/{B}] resampling blocks...")

        choice = rng.integers(0, Nb, size=Nb)
        idx = np.concatenate([idx_lists[i] for i in choice])

        a_sel = aO[idx]
        n_sel = n[idx]
        sigma_sel = sigma_gal[idx] if sigma_gal is not None else None

        # --- lensing / IA injection ---
        if lensing:
            p = int(rng.integers(0, npix_dir))
            d_hat = np.array(hp.pix2vec(lensing_nside_dir, p, nest=True))
            d_hat = d_hat / np.linalg.norm(d_hat)
            a_sel = _rotate_tangent_toward_dir(a_sel, n_sel, d_hat, theta_L)

        # Maps of the subset (obs)
        res_obs = _compute_maps_from_arrays(
            a_sel,
            n_sel,
            sigma_gal=sigma_sel,
            error_map=error_map,
            nside=nside_eval,
            half_sky=half_sky,
            nest=nest,
        )

        if with_psf:
            # Maps of the subset (PSF) -- no lensing injection
            res_psf = _compute_maps_from_arrays(
                aP[idx],
                n_sel,
                sigma_gal=sigma_sel,
                error_map=error_map,
                nside=nside_eval,
                half_sky=half_sky,
                nest=nest,
            )
            out = analyze_with_psf_corrections(
                res_obs["Ea"],
                res_obs["Eb"],
                res_obs["Eab"],
                res_psf["Ea"],
                res_psf["Eb"],
                res_psf["Eab"],
                nside=res_obs["nside"],
                c_bar=res_obs["c_bar"],
                nest=nest,
            )
            rows.append(
                {
                    "boot": b,
                    "nside": int(res_obs["nside"]),
                    "C_bar": float(res_obs["c_bar"]),
                    "eta_c_real": float(np.real(out["eta_complex"])),
                    "eta_c_imag": float(np.imag(out["eta_complex"])),
                    "eta_c_abs": float(np.abs(out["eta_complex"])),
                    "eta_c_arg_deg": float(
                        np.degrees(np.angle(out["eta_complex"]))
                    )
                    if np.abs(out["eta_complex"]) > 0
                    else 0.0,
                    "D_pix": out["D_IA"]["pix"],
                    "D_ra_deg": out["D_IA"]["ra_deg"],
                    "D_dec_deg": out["D_IA"]["dec_deg"],
                    "A_IA": out["A_IA"],
                    "Eam_at_D": out["Eam_at_D"],
                    "Eam_over_W_at_D": out["Eam_over_W_at_D"],
                    "da_pix": out["d_IA_a"]["pix"],
                    "theta_a_arcmin": out["theta_IA_a_arcmin"],
                    "db_pix": out["d_IA_b"]["pix"],
                    "theta_b_arcmin": out["theta_IA_b_arcmin"],
                }
            )

            if save_residual_maps:
                base_name = (
                    f"{base}_boot{b:04d}_{mode}"
                    f"{'_fullsky' if not half_sky else ''}"
                )
                out_base = os.path.join(residual_dir, base_name)
                _save_map_fits(
                    out_base + "_Ea_res_c.fits", out["Ea_res_c"], nside=res_obs["nside"]
                )
                _save_map_fits(
                    out_base + "_Eb_res_c.fits", out["Eb_res_c"], nside=res_obs["nside"]
                )
                _save_map_fits(
                    out_base + "_Eam_res.fits", out["Eam_res"], nside=res_obs["nside"]
                )
                _save_map_fits(
                    out_base + "_Eab_res.fits", out["Eab_res"], nside=res_obs["nside"]
                )
                print(
                    f"[BOOT {b+1}/{B}] residual maps saved as '{out_base}_*.fits'"
                )
        else:
            out = analyze_dual_and_DIA(
                res_obs["Ea"],
                res_obs["Eb"],
                res_obs["Eab"],
                res_obs["Eam"],
                nside=res_obs["nside"],
                c_bar=res_obs["c_bar"],
                nest=nest,
            )
            rows.append(
                {
                    "boot": b,
                    "nside": int(res_obs["nside"]),
                    "C_bar": float(res_obs["c_bar"]),
                    "D_pix": out["D_IA"]["pix"],
                    "D_ra_deg": out["D_IA"]["ra_deg"],
                    "D_dec_deg": out["D_IA"]["dec_deg"],
                    "A_IA": out["A_IA"],
                    "Eam_at_D": out["Eam_at_D"],
                    "Eam_over_W_at_D": out["Eam_over_W_at_D"],
                    "da_pix": out["d_IA_a"]["pix"],
                    "theta_a_arcmin": out["theta_IA_a_arcmin"],
                    "db_pix": out["d_IA_b"]["pix"],
                    "theta_b_arcmin": out["theta_IA_b_arcmin"],
                }
            )

            if save_residual_maps:
                base_name = (
                    f"{base}_boot{b:04d}_{mode}"
                    f"{'_fullsky' if not half_sky else ''}"
                )
                out_base = os.path.join(residual_dir, base_name)
                _save_map_fits(
                    out_base + "_Ea.fits", res_obs["Ea"], nside=res_obs["nside"]
                )
                _save_map_fits(
                    out_base + "_Eb.fits", res_obs["Eb"], nside=res_obs["nside"]
                )
                _save_map_fits(
                    out_base + "_Eab.fits", res_obs["Eab"], nside=res_obs["nside"]
                )
                _save_map_fits(
                    out_base + "_Ea_minus_b.fits",
                    res_obs["Eam"],
                    nside=res_obs["nside"],
                )
                print(
                    f"[BOOT {b+1}/{B}] maps (Ea, Eb, Eab, Ea-b) "
                    f"saved as '{out_base}_*.fits'"
                )

        print(f"[BOOT {b+1}/{B}] done in {time.time() - t1_iter:.2f}s")

    tab = Table(rows=rows)
    out_fits = os.path.join(
        outdir,
        f"{base}_bootstrap_ns{nside_blocks}_B{B}_{mode}"
        f"{'_fullsky' if not half_sky else ''}.fits",
    )
    tab.write(out_fits, format="fits", overwrite=True)
    print(
        f"[BOOT-{mode}] saved: {out_fits} "
        f"(B={B}, NSIDE_blocks={nside_blocks}, nside_eval={nside_eval}, "
        f"half_sky={half_sky})"
    )
    print(f"[BOOT] total elapsed: {time.time() - t0_boot:.2f}s")
    return out_fits


def _vector_from_pix(nside, ipix, nest=True):
    """
    Return the unit vector corresponding to a HEALPix pixel index.

    Parameters
    ----------
    nside : int
        NSIDE of the HEALPix grid.
    ipix : int
        Pixel index.
    nest : bool, optional
        If True, use NESTED ordering.

    Returns
    -------
    v : ndarray, shape (3,)
        Unit vector (x, y, z).
    """
    x, y, z = hp.pix2vec(nside, int(ipix), nest=nest)
    v = np.array([x, y, z], dtype=float)
    return v / np.linalg.norm(v)


def aggregate_bootstrap_results_for_all_sims(
    output_dir,
    *,
    prefix="simulation_d_xyz_t_0_sim_",
    sims=range(10),
    boot_subdir="boot",
    mode="nopsf",
    nest=True,
):
    """
    Aggregate bootstrap results over all simulations.

    For each simulation, reads all files matching:
        <output_dir>/<boot_subdir>/<prefix><sim>_bootstrap_ns*_B*_{mode}*.fits

    and computes:
        - median, p16, p84, p2.5, p97.5 for A_IA, theta_a, theta_b
        - mean direction of D_IA (RA/Dec) and angular scatter r68/r95

    Parameters
    ----------
    output_dir : str
        Base directory where bootstrap results are stored.
    prefix : str, optional
        Filename prefix for simulations.
    sims : iterable, optional
        Iterable of simulation indices (same as in batch_analyze).
    boot_subdir : str, optional
        Subdirectory inside output_dir containing bootstrap FITS files.
    mode : str, optional
        Mode suffix ("nopsf", "withpsf", "withpsf_lens", etc.).
    nest : bool, optional
        Whether to use NESTED ordering when reconstructing vectors.

    Returns
    -------
    out_tab : astropy.table.Table or None
        Summary table written to disk and returned. If no rows are
        produced, returns None.
    """
    rows = []
    for j in sims:
        pattern = os.path.join(
            output_dir,
            boot_subdir,
            f"{prefix}{j}_bootstrap_ns*_B*_{mode}*.fits",
        )
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[BOOT-AGG {mode}] no files found for simulation {j}")
            continue

        all_tabs = [Table.read(f) for f in files]
        # Concatenate by rows
        T = vstack(all_tabs, metadata_conflicts="silent")

        nside = int(np.unique(T["nside"])[0])
        A = np.array(T["A_IA"], float)
        th_a = np.array(T["theta_a_arcmin"], float)
        th_b = np.array(T["theta_b_arcmin"], float)

        v_list = np.stack(
            [_vector_from_pix(nside, ip, nest=nest) for ip in T["D_pix"]], axis=0
        )
        v_mean = v_list.mean(axis=0)
        v_mean /= np.linalg.norm(v_mean) + 1e-30
        ang = np.rad2deg(
            np.arccos(np.clip(v_list @ v_mean, -1.0, 1.0))
        )
        r68 = float(np.percentile(ang, 68.27))
        r95 = float(np.percentile(ang, 95.45))
        ra_m = float(np.degrees(np.arctan2(v_mean[1], v_mean[0]))) % 360
        dec_m = float(np.degrees(np.arcsin(v_mean[2])))

        def pct(x, p):
            return float(np.percentile(x, p))

        row = {
            "sim": j,
            "nside": nside,
            # A_IA
            "A_med": float(np.median(A)),
            "A_p16": pct(A, 16),
            "A_p84": pct(A, 84),
            "A_p2p5": pct(A, 2.5),
            "A_p97p5": pct(A, 97.5),
            # theta_a
            "theta_a_med": float(np.median(th_a)),
            "theta_a_p16": pct(th_a, 16),
            "theta_a_p84": pct(th_a, 84),
            "theta_a_p2p5": pct(th_a, 2.5),
            "theta_a_p97p5": pct(th_a, 97.5),
            # theta_b
            "theta_b_med": float(np.median(th_b)),
            "theta_b_p16": pct(th_b, 16),
            "theta_b_p84": pct(th_b, 84),
            "theta_b_p2p5": pct(th_b, 2.5),
            "theta_b_p97p5": pct(th_b, 97.5),
            # direction
            "D_ra_med": ra_m,
            "D_dec_med": dec_m,
            "D_r68_deg": r68,
            "D_r95_deg": r95,
            # count
            "N_boot": int(len(T)),
            # mode
            "mode": mode,
        }
        rows.append(row)

    if not rows:
        print(f"[BOOT-AGG {mode}] no summary generated.")
        return None

    out_tab = Table(rows=rows)
    out_path = os.path.join(output_dir, f"summary_bootstrap_{mode}.fits")
    out_tab.write(out_path, format="fits", overwrite=True)
    print(f"[BOOT-AGG {mode}] summary saved: {out_path}")
    return out_tab


# ===================== MAIN (minimal example) =====================


if __name__ == "__main__":
    # ==================== observational configuration ====================
    # NOTE:
    #   Replace these placeholders with your own directories and catalog names.
    #   - `input_dir` should contain the observed and PSF catalogs (FITS files).
    #   - `output_dir` is where maps and analysis products will be written.

    input_dir = "path/to/observed_catalogs"      # directory with observed + PSF catalogs
    output_dir = "path/to/output_directory"      # directory where results will be saved
    os.makedirs(output_dir, exist_ok=True)

    # Base names (without extension) for the observed and PSF catalogs.
    # Example: if your observed catalog is "my_obs_catalog.fits", use obs_name = "my_obs_catalog".
    obs_name = "observed_catalog"                # observed catalog base name (no extension)
    psf_name = "observed_catalog_psf"            # PSF catalog base name (no extension)

    arquivo_fits_obs = os.path.join(input_dir, f"{obs_name}.fits")
    arquivo_fits_psf = os.path.join(input_dir, f"{psf_name}.fits")

    # Bootstrap (with PSF and with additional lensing-like signal)
    DO_BOOTSTRAP_WITHPSF_LENS = True
    BOOT_B = 500
    BOOT_SUBDIR = "boot_obs"

    # Evaluate northern hemisphere first (z >= 0)
    HALF_SKY = True

    # Use per-galaxy sigma (column 'sigma'); no error_map needed here
    error_map = None

    # ==================== estimator: maps for OBS and PSF ====================
    t0_case = time.time()
    print("[OBS] Starting estimator (obs + psf)")

    # Observed
    run_estimator_new(
        arquivo_fits_obs,
        error_map=error_map,  # None -> use 'sigma' column
        nside=32,
        show_plot=False,
        output_dir=output_dir,
        sigma_col="sigma",    # per-galaxy sigma in radians
        half_sky=HALF_SKY,    # northern hemisphere (taking advantage of antipodal symmetry)
        nest=True,
    )

    # PSF
    run_estimator_new(
        arquivo_fits_psf,
        error_map=error_map,  # None -> use 'sigma' from PSF catalog
        nside=32,
        show_plot=False,
        output_dir=output_dir,
        sigma_col="sigma",
        half_sky=HALF_SKY,
        nest=True,
    )

    # ==================== direct measurement (with PSF correction) ====================
    print("[OBS] Direct measurement with PSF correction")

    base_obs = os.path.join(output_dir, obs_name)
    base_psf = os.path.join(output_dir, psf_name)

    p_Ea_obs = base_obs + "_Ea.fits"
    p_Eb_obs = base_obs + "_Eb.fits"
    p_Eab_obs = base_obs + "_Eab.fits"
    p_stat = base_obs + "_stats.txt"
    p_Ea_psf = base_psf + "_Ea.fits"
    p_Eb_psf = base_psf + "_Eb.fits"
    p_Eab_psf = base_psf + "_Eab.fits"

    Ea_obs = fits.getdata(p_Ea_obs)
    Eb_obs = fits.getdata(p_Eb_obs)
    Eab_obs = fits.getdata(p_Eab_obs)
    Ea_psf = fits.getdata(p_Ea_psf)
    Eb_psf = fits.getdata(p_Eb_psf)
    Eab_psf = fits.getdata(p_Eab_psf)

    # NSIDE of the maps
    try:
        hdr = fits.getheader(p_Ea_obs)
        nside_maps = int(hdr.get("NSIDE", 32))
    except Exception:
        nside_maps = 32

    # C_bar and stats (from observed catalog)
    Cbar, stats = _read_cbar(p_stat)

    # PSF correction with baseline de-leak: real=0.5, complex=0.0
    res_direct = analyze_with_psf_corrections(
        Ea_obs,
        Eb_obs,
        Eab_obs,
        Ea_psf,
        Eb_psf,
        Eab_psf,
        nside=nside_maps,
        c_bar=Cbar,
        nest=True,
        de_leak_mode="baseline",
        baseline_real=0.5,
        baseline_complex=0.0,
    )

    # Save a one-row FITS table with the direct measurement (no bootstrap)
    row = {
        "sim": -1,
        "nside": nside_maps,
        "C_bar": Cbar,
        "eta_c_real": float(np.real(res_direct["eta_complex"])),
        "eta_c_imag": float(np.imag(res_direct["eta_complex"])),
        "eta_c_abs": float(np.abs(res_direct["eta_complex"])),
        "eta_c_arg_deg": float(
            np.degrees(np.angle(res_direct["eta_complex"]))
        )
        if np.abs(res_direct["eta_complex"]) > 0
        else 0.0,
        "D_pix": res_direct["D_IA"]["pix"],
        "D_ra_deg": res_direct["D_IA"]["ra_deg"],
        "D_dec_deg": res_direct["D_IA"]["dec_deg"],
        "A_IA": res_direct["A_IA"],
        "Eam_at_D": res_direct["Eam_at_D"],
        "Eam_over_W_at_D": res_direct["Eam_over_W_at_D"],
        "eta_a": res_direct["eta_a"],
        "eta_b": res_direct["eta_b"],
        "da_pix": res_direct["d_IA_a"]["pix"],
        "theta_a_arcmin": res_direct["theta_IA_a_arcmin"],
        "db_pix": res_direct["d_IA_b"]["pix"],
        "theta_b_arcmin": res_direct["theta_IA_b_arcmin"],
        # coordinates of d_IA for a and b
        "da_ra_deg": res_direct["d_IA_a"]["ra_deg"],
        "da_dec_deg": res_direct["d_IA_a"]["dec_deg"],
        "da_l_deg": res_direct["d_IA_a"]["l_deg"],
        "da_b_deg": res_direct["d_IA_a"]["b_deg"],
        "db_ra_deg": res_direct["d_IA_b"]["ra_deg"],
        "db_dec_deg": res_direct["d_IA_b"]["dec_deg"],
        "db_l_deg": res_direct["d_IA_b"]["l_deg"],
        "db_b_deg": res_direct["d_IA_b"]["b_deg"],
    }
    _save_result_fits_per_sim(
        base_obs,
        row,
        nside=nside_maps,
        nest=True,
        cbar=Cbar,
        stats=stats,
        suffix="obs_psf",
    )

    # ==================== bootstrap with PSF + lensing-like signal ====================
    if DO_BOOTSTRAP_WITHPSF_LENS:
        print("[OBS] Bootstrap with PSF + additional lensing-like signal (1')")

        _ = bootstrap_errors_from_catalog(
            arquivo_fits=arquivo_fits_obs,
            error_map=error_map,  # None -> per-galaxy sigma
            arquivo_fits_psf=arquivo_fits_psf,
            with_psf=True,
            lensing=True,
            lensing_theta_arcmin=1.0,
            lensing_nside_dir=4096,
            B=BOOT_B,
            nside_blocks=16,
            nside_eval=nside_maps,
            half_sky=HALF_SKY,
            nest=True,
            outdir=os.path.join(output_dir, BOOT_SUBDIR),
            seed=1337,
            sigma_col="sigma",
            save_residual_maps=False,
        )

        # Aggregate bootstrap results (mode: withpsf_lens)
        _ = aggregate_bootstrap_results_for_all_sims(
            output_dir,
            prefix=f"{obs_name}",
            sims=[""],
            boot_subdir=BOOT_SUBDIR,
            mode="withpsf_lens",
            nest=True,
        )

    print(f"[OBS] Total elapsed: {time.time() - t0_case:.2f}s")

