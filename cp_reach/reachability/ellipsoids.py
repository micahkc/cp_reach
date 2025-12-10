import numpy as np

def ellipsoid_bounds_and_radius(P_scaled, r, point_sampler):
    """
    Given a scaled ellipsoid {x : x^T P_scaled x <= 1}, compute:
      - sampled boundary points
      - axis-aligned bounds (dim, 2)
      - conservative infinity-norm overapproximation r * max ||R[i]||
    
    Parameters
    ----------
    P_scaled : (n, n) ndarray
        Scaled Lyapunov matrix defining the ellipsoid.
    r : float
        Scaling factor sqrt(val) used for the infinity-norm bound.
    point_sampler : callable
        Function that generates boundary sample points given P_scaled,
        e.g. angular_acceleration.obtain_points.
    
    Returns
    -------
    points : (n, N) ndarray
        Sampled ellipsoid boundary points.
    bounds : (n, 2) ndarray
        Axis-aligned bounds, bounds[:,0]=lower, bounds[:,1]=upper.
    radius_inf : float
        Infinity-norm overapproximation of the ellipsoid.
    """
    # Eigen-decomposition (assumes symmetric PD)
    eigvals, eigvecs = np.linalg.eigh(P_scaled)
    eigvals = np.maximum(eigvals, 1e-12)   # ensure positive
    R = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))

    # Sample ellipsoid points
    points = point_sampler(P_scaled)

    # Axis-aligned bounding box
    lo = np.min(points, axis=1)
    hi = np.max(points, axis=1)
    bounds = np.stack([lo, hi], axis=1)

    # Infinity-norm conservative radius
    radius_inf = r * np.max(np.linalg.norm(R, axis=1))

    return points, bounds, radius_inf
