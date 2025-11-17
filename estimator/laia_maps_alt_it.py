import os
import time
import healpy as hp
import numpy as np
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import numexpr as ne
import math
from scipy.optimize import brentq, newton


def recover_alpha(m, c_bar, *, tol=1e-12, maxiter=4096, eps=1e-9):
    """
    Solve for alpha in

        0.5 + (c_bar / pi) * (alpha + 0.5 * sin(2 alpha)) = m

    with alpha in [0, pi/2].

    Strategy:
      1. Clip m into the physical domain (0.5, 0.5 + c_bar/2).
      2. Try a brentq root-finding on [0, pi/2] (robust).
      3. If the function does not change sign in that interval,
         fall back to Newton–Raphson starting from alpha = 0.

    Parameters
    ----------
    m : float
        Measured mean value of the statistic.
    c_bar : float
        Effective concentration parameter (C̄).
    tol : float, optional
        Absolute and relative tolerance for the solvers.
    maxiter : int, optional
        Maximum number of iterations for the solvers.
    eps : float, optional
        Small safety margin for clipping m.

    Returns
    -------
    alpha : float
        Solution in radians (0 <= alpha <= pi/2).
    """

    # 1) Physical domain for m
    m_min = 0.5
    m_max = 0.5 + 0.5 * c_bar
    m_safe = np.clip(m, m_min + eps, m_max - eps)

    # 2) Function whose root we want
    def f(a):
        return 0.5 + (c_bar / math.pi) * (a + 0.5 * math.sin(2 * a)) - m_safe

    # 3) First try brentq on [0, pi/2]
    try:
        return brentq(
            f,
            a=0.0,
            b=math.pi / 2,
            xtol=tol,
            rtol=tol,
            maxiter=maxiter,
        )
    except ValueError:
        # No sign change in the interval -> fallback to Newton
        def fp(a):
            return (c_bar / math.pi) * (1.0 + math.cos(2 * a))

        return newton(f, x0=0.0, fprime=fp, tol=tol, maxiter=maxiter)


def compute_chi2_for_pixels(nside, pixel_indices, g, n, error_map, chunk_size):
    """
    Compute a chi^2-like map for a given set of HEALPix pixels at a given nside,
    using orientation vectors g and line-of-sight vectors n, processed in chunks.

    This function is the high-resolution counterpart used in the iterative
    refinement: it recomputes the chi^2 for a restricted group of pixels.

    Parameters
    ----------
    nside : int
        HEALPix NSIDE resolution (NESTED scheme).
    pixel_indices : np.ndarray
        Array of pixel indices (NESTED ordering).
    g : (N, 3) np.ndarray
        Tangent-plane orientation vectors for each object (e.g. g_rot).
    n : (N, 3) np.ndarray
        Line-of-sight unit vectors for each object.
    error_map : np.ndarray
        HEALPix map (NESTED) with the orientation error sigma for each pixel
        on the sky. For each galaxy we pick the sigma of the pixel where it lies.
    chunk_size : int
        Number of galaxies processed per chunk.

    Returns
    -------
    chi2 : np.ndarray
        Chi^2 values for the requested pixels (same length as pixel_indices).
    """
    # For each galaxy, get sigma from the error map at its sky position
    pix_err = hp.vec2pix(hp.npix2nside(error_map.size), n[:, 0], n[:, 1], n[:, 2], nest=True)
    sigma = error_map[pix_err]  # typical error in the sky direction where the galaxy lies
    inv_var = 1.0 / sigma**2
    weights = inv_var / inv_var.sum()

    # Pixel center vectors for the selected indices
    v = np.column_stack(hp.pix2vec(nside, pixel_indices, nest=True))
    chi2 = np.zeros(len(pixel_indices), dtype=np.float64)

    total_N = g.shape[0]
    for i in range(0, total_N, chunk_size):
        end_idx = min(i + chunk_size, total_N)
        n_chunk = n[i:end_idx]
        g_chunk = g[i:end_idx]
        w_chunk = weights[i:end_idx]

        # Dot products between pixel directions v and line-of-sight n
        dot_vn = np.dot(v, n_chunk.T)  # (n_pixels, chunk)
        denom = np.sqrt(np.maximum(1e-12, 1 - dot_vn**2))

        # Dot products between pixel directions v and orientation g
        dot_vg = np.dot(v, g_chunk.T)  # (n_pixels, chunk)
        dot_ng = np.sum(n_chunk * g_chunk, axis=1)  # (chunk,)

        # General expression for the tangent-projected orientation;
        # here g is already perpendicular to n, so n·g = 0.
        expr = "(((dot_vg - dot_vn * dot_ng) / denom)**2) * w"
        dot_d2 = ne.evaluate(
            expr,
            local_dict={
                "dot_vg": dot_vg,
                "dot_vn": dot_vn,
                "dot_ng": dot_ng[None, :],  # shape (1, chunk)
                "denom": denom,
                "w": w_chunk[None, :],  # shape (1, chunk)
            },
        )

        chi2 += np.sum(dot_d2, axis=1)

    return chi2


def run_estimator_g_rot(
    arquivo_fits,
    error_map,
    nside1=32,
    nside2=1024,
    nside3=4096,
    nside4=16384,
    nside5=32768,
    chunk_size=50000,
    show_plot=True,
    output_dir="./output_g_rot",
):
    """
    Run the g_rot-based estimator for a simulation catalog, with an
    iterative HEALPix refinement of the preferred direction.

    This is designed to explore how the maximum of the chi^2-like field
    behaves as we increase NSIDE. The logic is:

      1. Compute chi^2 on a full-sky HEALPix grid with NSIDE = nside1.
      2. Find the pixel with maximum chi^2 and its 8 neighbors.
      3. For each finer NSIDE (nside2, nside3, nside4, nside5):
         - Take the pixel group from the previous NSIDE.
         - Subdivide each pixel into its subpixels at the new NSIDE (NESTED).
         - Recompute chi^2 only on those subpixels.
         - Update the maximum and its neighbors.
         - Store the result.

    Parameters
    ----------
    arquivo_fits : str
        Path to the input simulation catalog in FITS format.
        The table must contain columns:
          - g_rot_x, g_rot_y, g_rot_z: rotated tangent-plane orientations
          - n_x, n_y, n_z: line-of-sight unit vectors
    error_map : np.ndarray
        HEALPix map (NESTED) with per-pixel orientation errors (sigma).
    nside1, nside2, nside3, nside4, nside5 : int
        HEALPix NSIDE resolutions for the iterative refinement.
    chunk_size : int, optional
        Number of galaxies processed per chunk (default: 50_000).
    show_plot : bool, optional
        If True, display the chi^2 map at NSIDE = nside1.
    output_dir : str, optional
        Output directory where maps and iterative results are saved.

    Returns
    -------
    chi2_map_rot_full : np.ndarray
        Full-sky chi^2 map at NSIDE = nside1 (NESTED).
    iterative_results : list of dict
        One entry per NSIDE, with keys:
            - 'nside' : NSIDE used
            - 'max_pixel' : pixel index of the maximum chi^2
            - 'neighbors' : list of that pixel's neighbors
            - 'max_value' : maximum chi^2 value
            - 'theta_ia' : alpha (in radians) obtained via recover_alpha
    """
    t0 = time.time()

    tabela = Table.read(arquivo_fits)

    # Extract g_rot and n vectors directly from the table
    g = np.column_stack((tabela["g_rot_x"], tabela["g_rot_y"], tabela["g_rot_z"]))
    n = np.column_stack((tabela["n_x"], tabela["n_y"], tabela["n_z"]))

    # Per-galaxy sigma from the error map
    pix_err = hp.vec2pix(hp.npix2nside(error_map.size), n[:, 0], n[:, 1], n[:, 2], nest=True)
    sigma = error_map[pix_err]  # typical error in the sky direction where the galaxy lies
    inv_var = 1.0 / sigma**2
    weights = inv_var / inv_var.sum()

    # Effective concentration C̄
    c = np.exp(-2 * (sigma**2))
    c_bar = np.sum(weights * c)

    # ----- NSIDE = nside1: build full-sky chi^2 map (northern hemisphere only) -----

    npix_total = hp.nside2npix(nside1)
    ipix = np.arange(npix_total)
    x, y, z = hp.pix2vec(nside1, ipix, nest=True)
    v = np.vstack((x, y, z)).T  # (npix_total, 3)

    # Restrict to northern hemisphere (z >= 0)
    mask = v[:, 2] >= 0
    v = v[mask]
    npix = np.sum(mask)

    chi2_map_rot = np.zeros(npix, dtype=np.float64)

    total_N = g.shape[0]
    for i in range(0, total_N, chunk_size):
        end_idx = min(i + chunk_size, total_N)
        n_chunk = n[i:end_idx]  # (chunk, 3)
        g_chunk = g[i:end_idx]  # (chunk, 3)
        w_chunk = weights[i:end_idx]

        # dot(v, n)
        dot_vn = np.dot(v, n_chunk.T)  # (npix, chunk)
        denom = np.sqrt(np.maximum(1e-12, 1 - dot_vn**2))

        # dot(v, g)
        dot_vg = np.dot(v, g_chunk.T)  # (npix, chunk)

        # g is already perpendicular to n, so n·g = 0 and:
        # cos(theta) = (v·g) / sqrt(1 - (v·n)^2)
        expr = "((dot_vg / denom)**2)*w"
        dot_d2 = ne.evaluate(
            expr,
            local_dict={
                "dot_vg": dot_vg,
                "denom": denom,
                "w": w_chunk[None, :],  # (1, chunk)
            },
        )

        # Accumulate contribution to chi^2 map
        chi2_map_rot += np.sum(dot_d2, axis=1)

        current_chunk = i // chunk_size + 1
        num_chunks = np.ceil(total_N / chunk_size)
        elapsed = time.time() - t0
        percent = 100 * current_chunk / num_chunks
        print(
            f"Chunk {current_chunk}/{int(num_chunks)} "
            f"({percent:.1f}%) done. Elapsed time: {elapsed:.2f} s"
        )

    # Reconstruct full map (including southern hemisphere, set to zero here)
    chi2_map_rot_full = np.zeros(npix_total)
    chi2_map_rot_full[mask] = chi2_map_rot

    if show_plot:
        hp.mollview(
            chi2_map_rot_full,
            title="Chi^2 map (g_rot)",
            unit="Chi^2",
            cmap="viridis",
            nest=True,
        )
        plt.show()

    tempo_total = time.time() - t0
    print(f"Total time (NSIDE={nside1} map): {tempo_total:.2f} seconds.")

    # Save chi^2 map at NSIDE = nside1
    hdu = fits.PrimaryHDU(chi2_map_rot_full)
    header = hdu.header
    header["NSIDE"] = nside1
    header["CHKSIZE"] = chunk_size
    header["SIMFILE"] = os.path.basename(arquivo_fits)
    header["VECTOR"] = "g_rot"
    header["EXECTIME"] = tempo_total

    base = os.path.basename(arquivo_fits)
    name, ext = os.path.splitext(base)
    output_file = os.path.join(output_dir, f"{name}_g_rot_chi2.fits")
    hdu.writeto(output_file, overwrite=True)
    print(f"g_rot chi^2 map saved to {output_file}")

    # ===== Iterative refinement over multiple NSIDEs =====
    iterative_results = []

    # Step 1: refinement at NSIDE = nside1 (already computed)
    nside_current = nside1
    npix_current = hp.nside2npix(nside_current)
    all_pixels = np.arange(npix_current)

    # No mask here: use all pixels
    group_current = all_pixels
    chi2_current = chi2_map_rot_full[group_current]
    max_idx = np.argmax(chi2_current)
    max_pix = group_current[max_idx]
    neighbors = hp.get_all_neighbours(nside_current, max_pix, nest=True)
    neighbors = neighbors[neighbors >= 0]
    group_iter = np.concatenate(([max_pix], neighbors))
    max_value = chi2_map_rot_full[max_pix]
    theta_ia = recover_alpha(max_value, c_bar)

    iterative_results.append(
        {
            "nside": nside_current,
            "max_pixel": int(max_pix),
            "neighbors": group_iter.tolist(),
            "max_value": float(max_value),
            "theta_ia": float(theta_ia),
        }
    )

    # Steps 2–5: refinement for nside2, nside3, nside4, nside5
    nsides_list = [nside2, nside3, nside4, nside5]
    current_group = group_iter  # group of pixels selected at the previous iteration
    nside_prev = nside_current

    for nside_new in nsides_list:
        # Subdivide pixels from nside_prev to nside_new (NESTED):
        # each pixel splits into (factor^2) subpixels.
        factor = nside_new // nside_prev
        new_group = []
        for pix in current_group:
            subpix = [pix * (factor**2) + i for i in range(factor**2)]
            new_group.extend(subpix)
        new_group = np.array(new_group)

        # Compute chi^2 for the new group at nside_new
        chi2_new = compute_chi2_for_pixels(nside_new, new_group, g, n, error_map, chunk_size)
        max_idx = np.argmax(chi2_new)
        max_pix_new = new_group[max_idx]
        neighbors_new = hp.get_all_neighbours(nside_new, max_pix_new, nest=True)
        neighbors_new = neighbors_new[neighbors_new >= 0]
        group_new_iter = np.concatenate(([max_pix_new], neighbors_new))
        max_value_new = chi2_new[max_idx]
        theta_ia = recover_alpha(max_value_new, c_bar)

        iterative_results.append(
            {
                "nside": nside_new,
                "max_pixel": int(max_pix_new),
                "neighbors": group_new_iter.tolist(),
                "max_value": float(max_value_new),
                "theta_ia": float(theta_ia),
            }
        )

        # Prepare for next iteration
        current_group = group_new_iter
        nside_prev = nside_new

        tempo_total = time.time() - t0
        print(f"Total time (up to NSIDE={nside_new}): {tempo_total:.2f} seconds.")

    # Save iterative results to a FITS table (one row per NSIDE)
    it_data = {
        "NSIDE": [res["nside"] for res in iterative_results],
        "MAX_PIXEL": [res["max_pixel"] for res in iterative_results],
        "MAX_VALUE": [res["max_value"] for res in iterative_results],
        "NEIGHBORS": [", ".join(map(str, res["neighbors"])) for res in iterative_results],
        "THETA_IA": [res["theta_ia"] for res in iterative_results],
    }
    it_table = Table(it_data)
    it_output_file = os.path.join(output_dir, f"{name}_g_rot_iterative_results.fits")
    it_table.write(it_output_file, overwrite=True)
    print(f"Iterative g_rot results saved to {it_output_file}")

    tempo_total = time.time() - t0
    print(f"Total time (full run): {tempo_total:.2f} seconds.")

    # Return chi^2 map at NSIDE = nside1 and full iterative chain
    return chi2_map_rot_full, iterative_results


# ===== Example usage / driver loop =====
# NOTE: keep these as placeholders. Replace with your own paths and settings.

# Directory containing the simulation FITS catalogs, e.g.
# simulation_d_xyz_t_0_sim_0.fits, simulation_d_xyz_t_0_sim_1.fits, ...
input_dir = "path/to/simulation_catalogs"

# Output directory where chi^2 maps and iterative results will be saved
output_dir = "path/to/output_directory"

# HEALPix error map (NESTED) with per-pixel orientation errors
error_map = hp.read_map("path/to/shape_catalog_error.fits", nest=True)

resultados_g_rot = []

# Example loops:
# - outer loops over "direction d" and "theta index" are kept for compatibility
#   with earlier experiments, but use range(1) here (single value) by default.
for i in range(1):  # loop over LAIA directions (placeholder)
    for t_idx in range(1):  # loop over theta values (placeholder)
        for j in range(128):  # loop over simulations (adjust as needed)
            arquivo_fits = os.path.join(input_dir, f"simulation_d_xyz_t_0_sim_{j}.fits")
            chi2_map_rot, iterative_results = run_estimator_g_rot(
                arquivo_fits,
                error_map,
                nside1=32,
                nside2=1024,
                nside3=4096,
                nside4=16384,
                nside5=32768,
                chunk_size=50000,
                show_plot=False,
                output_dir=output_dir,
            )
            print(f"Estimator run for: {arquivo_fits}")
            resultados_g_rot.append((chi2_map_rot, iterative_results))

