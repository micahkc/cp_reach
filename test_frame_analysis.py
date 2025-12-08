"""
Analyze what frame the gravity feedforward should be in.

The log-linear dynamics are: ξ̇ = -ad_n̄ ξ + A_C ξ + b(t)

where ξ is the Lie algebra error (body frame representation).

The coupling matrix A_C has the structure:
ξ̇_p = ξ_v
ξ̇_v = 0
ξ̇_R = 0

So without feedforward, velocity error doesn't change (constant).

The gravity feedforward b(t) = [0, g_diff, 0] adds acceleration to velocity error.

Question: Should g_diff be in inertial frame or body frame?

Answer: Since ξ_v represents velocity error in the body frame (for right-invariant error),
the feedforward should ALSO be in the body frame!

So we need: b = [0, R_ref^T @ (g(p_actual) - g(p_ref)), 0]
"""
print(__doc__)
