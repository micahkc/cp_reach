"""
Visualization utilities for log-linear error dynamics comparison.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_error_comparison(results, save_path=None):
    """
    Create comprehensive comparison plots.

    Parameters
    ----------
    results : dict
        Results from SimulationComparison.simulate_both()
    save_path : str, optional
        Path to save figure. If None, displays interactively.
    """
    t = results['t']
    xi_nonlinear = results['xi_nonlinear']
    xi_loglinear = results['xi_loglinear']

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # Position error
    ax = axes[0]
    ax.plot(t, xi_nonlinear[:, 0], 'b-', linewidth=2, label='Nonlinear (ξₚₓ)')
    ax.plot(t, xi_loglinear[:, 0], 'r--', linewidth=1.5, label='Log-linear (ξₚₓ)')
    ax.plot(t, xi_nonlinear[:, 1], 'g-', linewidth=2, alpha=0.7, label='Nonlinear (ξₚᵧ)')
    ax.plot(t, xi_loglinear[:, 1], 'm--', linewidth=1.5, alpha=0.7, label='Log-linear (ξₚᵧ)')
    ax.set_ylabel('Position Error ξₚ (log-coords)', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Log-Linear vs Nonlinear Error: Exact Match', fontsize=13, fontweight='bold')

    # Velocity error
    ax = axes[1]
    ax.plot(t, xi_nonlinear[:, 3], 'b-', linewidth=2, label='Nonlinear (ξᵥₓ)')
    ax.plot(t, xi_loglinear[:, 3], 'r--', linewidth=1.5, label='Log-linear (ξᵥₓ)')
    ax.plot(t, xi_nonlinear[:, 4], 'g-', linewidth=2, alpha=0.7, label='Nonlinear (ξᵥᵧ)')
    ax.plot(t, xi_loglinear[:, 4], 'm--', linewidth=1.5, alpha=0.7, label='Log-linear (ξᵥᵧ)')
    ax.set_ylabel('Velocity Error ξᵥ (log-coords)', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Attitude error
    ax = axes[2]
    ax.plot(t, xi_nonlinear[:, 6], 'b-', linewidth=2, label='Nonlinear (ξᴿₓ)')
    ax.plot(t, xi_loglinear[:, 6], 'r--', linewidth=1.5, label='Log-linear (ξᴿₓ)')
    ax.plot(t, xi_nonlinear[:, 7], 'g-', linewidth=2, alpha=0.7, label='Nonlinear (ξᴿᵧ)')
    ax.plot(t, xi_loglinear[:, 7], 'm--', linewidth=1.5, alpha=0.7, label='Log-linear (ξᴿᵧ)')
    ax.set_ylabel('Attitude Error ξᴿ (log-coords)', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Error difference (log scale)
    ax = axes[3]
    error_diff = results['error_diff']
    ax.semilogy(t, error_diff, 'k-', linewidth=2)
    ax.set_ylabel('||ξ_nonlinear - ξ_loglinear||', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('Difference (Numerical Integration Error Only)', fontsize=10, style='italic')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    return fig


def plot_3d_trajectory(results, save_path=None):
    """
    Plot 3D trajectory of reference and actual spacecraft.

    Parameters
    ----------
    results : dict
        Results from SimulationComparison.simulate_both()
    save_path : str, optional
        Path to save figure.
    """
    state_ref = results['state_ref']
    state_actual = results['state_actual']

    p_ref = state_ref[:, 0:3] / 1e3  # convert to km
    p_actual = state_actual[:, 0:3] / 1e3

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2], 'b-', linewidth=2, label='Reference')
    ax.plot(p_actual[:, 0], p_actual[:, 1], p_actual[:, 2], 'r--', linewidth=1.5, label='Actual')

    # Mark initial positions
    ax.scatter(*p_ref[0, :], color='blue', s=100, marker='o', label='Ref Start')
    ax.scatter(*p_actual[0, :], color='red', s=100, marker='o', label='Actual Start')

    # Plot Earth
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    r_earth = 6371  # km
    x = r_earth * np.outer(np.cos(u), np.sin(v))
    y = r_earth * np.outer(np.sin(u), np.sin(v))
    z = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='cyan', alpha=0.3)

    ax.set_xlabel('X (km)', fontsize=11)
    ax.set_ylabel('Y (km)', fontsize=11)
    ax.set_zlabel('Z (km)', fontsize=11)
    ax.set_title('Spacecraft Trajectories in Inertial Frame', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    # Equal aspect ratio
    max_range = np.array([
        p_ref[:, 0].max() - p_ref[:, 0].min(),
        p_ref[:, 1].max() - p_ref[:, 1].min(),
        p_ref[:, 2].max() - p_ref[:, 2].min()
    ]).max() / 2.0

    mid_x = (p_ref[:, 0].max() + p_ref[:, 0].min()) * 0.5
    mid_y = (p_ref[:, 1].max() + p_ref[:, 1].min()) * 0.5
    mid_z = (p_ref[:, 2].max() + p_ref[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    return fig


def plot_all_components(results, save_path=None):
    """
    Plot all 9 components of the Lie algebra error.

    Parameters
    ----------
    results : dict
        Results from SimulationComparison.simulate_both()
    save_path : str, optional
        Path to save figure.
    """
    t = results['t']
    xi_nonlinear = results['xi_nonlinear']
    xi_loglinear = results['xi_loglinear']

    labels = [
        'ξₚₓ', 'ξₚᵧ', 'ξₚᵤ',
        'ξᵥₓ', 'ξᵥᵧ', 'ξᵥᵤ',
        'ξᴿₓ', 'ξᴿᵧ', 'ξᴿᵤ'
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]
        ax.plot(t, xi_nonlinear[:, i], 'b-', linewidth=2, label='Nonlinear')
        ax.plot(t, xi_loglinear[:, i], 'r--', linewidth=1.5, label='Log-linear')
        ax.set_ylabel(labels[i], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if i >= 6:
            ax.set_xlabel('Time (s)', fontsize=10)
        if i == 0:
            ax.legend(loc='best', fontsize=9)

    fig.suptitle('All 9 Components of Lie Algebra Error ξ ∈ se₂(3)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    return fig


def generate_paper_figure(results, save_path='simulation_comparison.pdf'):
    """
    Generate a figure similar to Figure 1 in the paper.

    Parameters
    ----------
    results : dict
        Results from SimulationComparison.simulate_both()
    save_path : str
        Path to save the figure
    """
    import matplotlib.pyplot as plt

    t = results['t']
    xi_nonlinear = results['xi_nonlinear']
    xi_loglinear = results['xi_loglinear']
    error_diff = results['error_diff']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot selected components
    components = [0, 3, 6]  # ξₚₓ, ξᵥₓ, ξᴿₓ
    labels_short = ['ξₚₓ (Position)', 'ξᵥₓ (Velocity)', 'ξᴿₓ (Attitude)']

    for i, (comp, label) in enumerate(zip(components, labels_short)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        ax.plot(t, xi_nonlinear[:, comp], 'b-', linewidth=2.5, label='From Nonlinear Sim')
        ax.plot(t, xi_loglinear[:, comp], 'r--', linewidth=2, label='Log-Linear Propagation')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Error plot
    ax = axes[1, 1]
    ax.semilogy(t, error_diff, 'k-', linewidth=2)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('||Δξ|| (Numerical Error)', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('Integration Error Only', fontsize=10, style='italic')

    fig.suptitle('Exact Log-Linear Error Dynamics on SE₂(3)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPaper-style figure saved to {save_path}")

    return fig


if __name__ == "__main__":
    print("This module provides visualization utilities.")
    print("Import and use with results from log_linear_dynamics.py")
