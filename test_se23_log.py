"""
Test to understand what ξ_p represents in SE₂(3) logarithmic coordinates.
"""
import numpy as np
import casadi as ca
import cyecca.lie as lie

# Create two close states
p_ref = np.array([0, -42164e3, 0])
v_ref = np.array([3070, 0, 0])
q_ref = np.array([1, 0, 0, 0])

# Small perturbation
delta_p = np.array([100, 50, 20])  # 100m difference
p_actual = p_ref + delta_p
v_actual = v_ref.copy()
q_actual = q_ref.copy()

# Convert to SE23Quat: [p, v, q]
X_ref = lie.SE23Quat.elem(ca.DM(np.concatenate([p_ref, v_ref, q_ref])))
X_actual = lie.SE23Quat.elem(ca.DM(np.concatenate([p_actual, v_actual, q_actual])))

# Compute group error
eta = X_ref.inverse() * X_actual

# Compute algebra error via log map
xi_lie = eta.log()
if hasattr(xi_lie.param, 'full'):
    xi = np.array(xi_lie.param.full()).flatten()
else:
    xi = np.array(ca.DM(xi_lie.param).full()).flatten()

print("=" * 80)
print("Understanding SE₂(3) Logarithmic Coordinates")
print("=" * 80)
print(f"\nPosition difference (Euclidean): {delta_p}")
print(f"Velocity difference (Euclidean): {v_actual - v_ref}")
print()
print(f"Log map result ξ:")
print(f"  ξ_p (position component): {xi[0:3]}")
print(f"  ξ_v (velocity component): {xi[3:6]}")
print(f"  ξ_R (attitude component): {xi[6:9]}")
print()
print(f"Difference between ξ_p and Euclidean delta:")
print(f"  ||ξ_p - delta_p|| = {np.linalg.norm(xi[0:3] - delta_p):.6e}")
print()

# Test inverse: can we recover the position from ξ_p?
print("Testing if ξ_p ≈ delta_p for small errors...")
if np.allclose(xi[0:3], delta_p, rtol=1e-6):
    print("✓ YES: For small errors, ξ_p ≈ delta_p (Euclidean difference)")
    print("  This means p_actual_est = p_ref + ξ_p is CORRECT!")
else:
    print("✗ NO: ξ_p is NOT simply the Euclidean position difference")
    print("  The log map introduces nonlinear transformations")
    print()
    print("Need to use exponential map to recover position:")

    # Try exponential map
    eta_reconstructed = lie.SE23Quat.elem(lie.se23.exp(xi_lie).param)

    # Extract position from reconstructed group element
    # X_actual_reconstructed = X_ref * eta_reconstructed
    print("  This requires proper inverse transformation through Exp map")

print("=" * 80)
