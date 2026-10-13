"""KML-free canonical orbit geometry from reconstructed camera poses.

All geometry stays in the COLMAP/3DGS similarity frame.  Metric scale, GPS and
true north are intentionally unnecessary: closure cameras only need to be in
the same frame as the trained Gaussian scene.
"""
from __future__ import annotations

import numpy as np


def _unit(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x)
    if not np.isfinite(n) or n < eps:
        raise ValueError('Cannot normalize a zero/non-finite vector')
    return x / n


def _camera_arrays(cameras):
    if len(cameras) < 3:
        raise ValueError('At least three reconstructed cameras are required')
    centers = np.stack([np.asarray(c['position'], dtype=float) for c in cameras])
    rotations = np.stack([np.asarray(c['rotation'], dtype=float) for c in cameras])
    if centers.shape[1:] != (3,) or rotations.shape[1:] != (3, 3):
        raise ValueError('Invalid camera array shapes')
    if not np.isfinite(centers).all() or not np.isfinite(rotations).all():
        raise ValueError('Non-finite reconstructed cameras')
    # Camera convention in this repository: camera-to-world columns are
    # right / down / forward, hence column 2 is the optical axis.
    forward = np.stack([_unit(q[:, 2]) for q in rotations])
    return centers, rotations, forward


def intersect_view_rays(centers, directions, iterations=8):
    """Robust least-squares point closest to all viewing *lines*.

    The line formulation makes the estimate insensitive to a global sign flip
    of a viewing axis.  IRLS downweights cameras with large perpendicular
    residuals, which is useful when a few COLMAP poses are poor.
    """
    centers = np.asarray(centers, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if centers.shape != directions.shape or centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError('centers/directions must both be Nx3')
    directions = np.stack([_unit(d) for d in directions])
    projectors = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    weights = np.ones(len(centers), dtype=float)
    target = centers.mean(0)
    residuals = np.zeros(len(centers), dtype=float)
    for _ in range(max(1, int(iterations))):
        A = np.einsum('n,nij->ij', weights, projectors)
        b = np.einsum('n,nij,nj->i', weights, projectors, centers)
        if np.linalg.cond(A) > 1e10:
            target = np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            target = np.linalg.solve(A, b)
        delta = target[None] - centers
        residuals = np.linalg.norm(np.einsum('nij,nj->ni', projectors, delta), axis=1)
        scale = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
        if not np.isfinite(scale) or scale < 1e-9:
            break
        cutoff = 1.5 * scale
        weights = np.where(residuals <= cutoff, 1.0, cutoff / np.maximum(residuals, 1e-12))
    if not np.isfinite(target).all():
        raise ValueError('Ray-intersection target is non-finite')
    return target, residuals, weights


def _canonical_basis(centers, target, forward):
    rel = centers - target[None]
    # Orbit plane normal is the least-variance direction after removing the
    # trajectory mean.  This handles circular or mildly spiral trajectories.
    centered = rel - rel.mean(0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered), 1)
    values, vectors = np.linalg.eigh(covariance)
    up = _unit(vectors[:, np.argmin(values)])

    # Choose the sign so cameras lie predominantly on the +up side of target.
    heights = rel @ up
    if np.median(heights) < 0:
        up = -up
        heights = -heights
    if abs(float(np.median(heights))) < 1e-7:
        # Fallback: cameras usually look roughly toward -up.
        if float(np.median(forward @ up)) > 0:
            up = -up
            heights = -heights

    # Heading zero is arbitrary by design.  Use the first well-conditioned
    # camera radial direction for deterministic per-scene canonicalization.
    e1 = None
    horizontal = rel - (rel @ up)[:, None] * up[None]
    order = np.argsort(-np.linalg.norm(horizontal, axis=1))
    for idx in order:
        if np.linalg.norm(horizontal[idx]) > 1e-8:
            e1 = _unit(horizontal[idx])
            break
    if e1 is None:
        raise ValueError('Camera trajectory has no horizontal extent')
    e2 = _unit(np.cross(up, e1))
    # e1 x e2 = up, a right-handed canonical world basis.
    if np.linalg.det(np.column_stack((e1, e2, up))) <= 0:
        raise AssertionError('Canonical orbit basis is not right-handed')
    return e1, e2, up, values


def estimate_orbit(cameras):
    """Estimate a canonical target-centered orbit from reconstructed cameras."""
    centers, rotations, forward = _camera_arrays(cameras)
    target, ray_residuals, ray_weights = intersect_view_rays(centers, forward)
    e1, e2, up, eigenvalues = _canonical_basis(centers, target, forward)
    rel = centers - target[None]
    slant = np.linalg.norm(rel, axis=1)
    height = rel @ up
    horizontal = np.sqrt(np.maximum(slant * slant - height * height, 0.0))
    tilt = np.rad2deg(np.arctan2(horizontal, height))
    view_to_target = np.stack([_unit(target - c) for c in centers])
    forward_cos = np.sum(view_to_target * forward, axis=1)
    if not np.isfinite(slant).all() or float(np.median(slant)) <= 0:
        raise ValueError('Invalid orbit radius')
    result = dict(
        schema='ocl_plus_canonical_orbit_v1',
        coordinate_frame='COLMAP/3DGS similarity frame; arbitrary scale/origin/yaw',
        target=target.tolist(),
        e1=e1.tolist(),
        e2=e2.tolist(),
        up=up.tolist(),
        median_slant_range=float(np.median(slant)),
        slant_range_q10_q50_q90=[float(x) for x in np.quantile(slant, [.1, .5, .9])],
        observed_tilt_q10_q50_q90=[float(x) for x in np.quantile(tilt, [.1, .5, .9])],
        ray_residual_q50_q90_max=[float(np.median(ray_residuals)),
                                  float(np.quantile(ray_residuals, .9)),
                                  float(np.max(ray_residuals))],
        forward_to_target_cos_q10_q50=[float(x) for x in np.quantile(forward_cos, [.1, .5])],
        plane_eigenvalues=[float(x) for x in eigenvalues],
        reconstructed_camera_count=len(cameras),
        robust_ray_weight_min=float(np.min(ray_weights)),
    )
    return result


def canonical_lookat(heading_deg, tilt_deg, distance, orbit, target=None):
    """Create a camera-to-world pose in the canonical orbit frame.

    ``tilt`` follows the original KML convention used by OCL: 0 degrees is
    straight down from above the target; 90 degrees is horizontal.
    """
    if not np.isfinite([heading_deg, tilt_deg, distance]).all() or distance <= 0:
        raise ValueError('Invalid closure LookAt parameters')
    if not 0 <= float(tilt_deg) <= 90:
        raise ValueError('Closure tilt must be in [0, 90] degrees')
    e1, e2, up = (_unit(orbit[k]) for k in ('e1', 'e2', 'up'))
    target = np.asarray(orbit['target'] if target is None else target, dtype=float)
    h, t = np.deg2rad([float(heading_deg), float(tilt_deg)])
    # Match kml_utils.lookat algebra in the canonical basis.
    horizontal_forward = np.sin(h) * e1 + np.cos(h) * e2
    right = np.cos(h) * e1 - np.sin(h) * e2
    forward = np.sin(t) * horizontal_forward - np.cos(t) * up
    forward, right = _unit(forward), _unit(right)
    down = _unit(np.cross(forward, right))
    Q = np.column_stack((right, down, forward))
    if not np.allclose(Q.T @ Q, np.eye(3), atol=1e-6) or np.linalg.det(Q) <= 0:
        raise ValueError('Generated closure rotation is invalid')
    center = target - float(distance) * forward
    return center, Q


def resolve_tilts(spec, orbit, count=4):
    """Resolve comma-separated degrees or ``auto`` to a deterministic list."""
    text = str(spec).strip().lower()
    if text != 'auto':
        values = [float(x.strip()) for x in str(spec).split(',') if x.strip()]
        if not values or any(not 0 <= x <= 90 for x in values):
            raise ValueError('--closure-tilts must be comma-separated degrees in [0,90] or auto')
        return values
    q10, q50, q90 = orbit['observed_tilt_q10_q50_q90']
    if q90 - q10 >= 6.0:
        # Spread four views across the observed elevation support.
        return [float(x) for x in np.linspace(q10, q90, count)]
    # Fixed-height circular datasets can have almost one observed tilt.  Add a
    # conservative symmetric elevation completion around it.
    offsets = np.linspace(-15.0, 15.0, count)
    values = np.clip(q50 + offsets, 1.0, 89.0)
    return [float(x) for x in values]
