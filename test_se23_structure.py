"""
Understand SE₂(3) right-invariant error structure.

SE₂(3) elements have the form:
X = [R  v  p]
    [0  R  0]
    [0  0  1]

Right-invariant error: η = X̄⁻¹ X

For position, this gives: p_error = R̄ᵀ(p - p̄)

So ξ_p is the position error expressed in the REFERENCE body frame!
To get inertial frame position: p_actual = p̄ + R̄ ξ_p
"""
import numpy as np
import casadi as ca
import cyecca.lie as lie

# Test case: reference with rotation
p_ref = np.array([0, -42164e3, 0])
v_ref = np.array([3070, 0, 0])
# 45° rotation about z-axis
angle = np.pi/4
q_ref = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])

# Actual state with simple position offset in INERTIAL frame
delta_p_inertial = np.array([100, 50, 20])
p_actual = p_ref + delta_p_inertial
v_actual = v_ref.copy()
q_actual = q_ref.copy()

# Create SE23 elements
X_ref = lie.SE23Quat.elem(ca.DM(np.concatenate([p_ref, v_ref, q_ref])))
X_actual = lie.SE23Quat.elem(ca.DM(np.concatenate([p_actual, v_actual, q_actual])))

# Compute right-invariant error
eta = X_ref.inverse() * X_actual
xi_lie = eta.log()
xi = np.array(ca.DM(xi_lie.param).full()).flatten()

# Get rotation matrix from quaternion
def quat_to_rot(q):
    """Convert quaternion [w,x,y,z] to rotation matrix"""
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
    ])

R_ref = quat_to_rot(q_ref)

# What should ξ_p be?
# For right-invariant error: ξ_p = R_ref^T @ delta_p_inertial
expected_xi_p = R_ref.T @ delta_p_inertial

print("=" * 80)
print("SE₂(3) Right-Invariant Error Structure")
print("=" * 80)
print(f"\nReference rotation angle: {angle*180/np.pi:.1f}°")
print(f"\nPosition offset (inertial frame): {delta_p_inertial}")
print(f"Expected ξ_p (body frame): {expected_xi_p}")
print(f"Actual ξ_p from log map: {xi[0:3]}")
print(f"\nDifference: {np.linalg.norm(xi[0:3] - expected_xi_p):.6e}")

# Test inverse transformation
print("\n" + "=" * 80)
print("Inverse Transformation")
print("=" * 80)
print(f"\nGiven ξ_p (body frame): {xi[0:3]}")
print(f"Transform to inertial: p_actual_est = p_ref + R_ref @ ξ_p")
p_actual_recovered = p_ref + R_ref @ xi[0:3]
print(f"  Recovered p_actual: {p_actual_recovered}")
print(f"  True p_actual: {p_actual}")
print(f"  Error: {np.linalg.norm(p_actual_recovered - p_actual):.6e}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("For right-invariant error on SE₂(3):")
print("  ξ_p is the position error in the REFERENCE BODY FRAME")
print("  To get inertial position: p_actual = p_ref + R_ref @ ξ_p")
print("  ")
print("Special case: When R_ref = I (identity), then ξ_p = delta_p_inertial")
print("  This is why our code works for the geostationary test case!")
print("=" * 80)
