from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
import numpy as np
import casadi as ca
import cyecca.lie as lie
import cp_reach.satellite.mission as sat_sim

class HandlerArrow(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        arrow = FancyArrowPatch(
            (0, 0),
            (20, 0),
            transform=trans,
            color='green',
            lw=2,
            arrowstyle='-|>,head_length=2,head_width=0.5',  # smaller head
            mutation_scale=fontsize * 0.5,                 # scale relative to font
            shrinkA=0, shrinkB=0
        )
        return [arrow]

def derive_angle_error():
    q_a = lie.SO3Quat.elem(ca.SX.sym('q_a', 4))
    q_b = lie.SO3Quat.elem(ca.SX.sym('q_b', 4))
    q_err = q_b.inverse() * q_a
    euler = lie.SO3EulerB321.from_Quat(q_err).param
    return ca.Function('f_ang_err', [q_a.param, q_b.param], [euler])

f_ang_err = derive_angle_error()


def plot_burn_angular_velocity(ax,data):
    sat = sat_sim.SatSimBurn()
    for i, r in enumerate(data):
        ang_err = []
        w_a = r['xf'][sat.model['x_index']['wx_a']:sat.model['x_index']['wz_a']+1, :].full().squeeze()
        w_b = r['xf'][sat.model['x_index']['wx_b']:sat.model['x_index']['wz_b']+1, :].full().squeeze()
        w_err = np.rad2deg(w_a - w_b)
        t = r['xf'][0, :].full().squeeze()
        h_x = ax.plot(t, w_err[0, :], 'r', alpha=0.2)
        h_y = ax.plot(t, w_err[1, :], 'g', alpha=0.2)
        h_z = ax.plot(t, w_err[2, :], 'b', alpha=0.2)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Ang. Vel. Error [deg/s]')

    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='x', alpha=0.5),
        Line2D([0], [0], color='green', lw=2, label='y', alpha=0.5),
        Line2D([0], [0], color='blue', lw=2, label='z', alpha=0.5),
    ]
    ax.legend(handles=legend_elements, ncol=3, loc='best')
    ax.grid()
    return ax

def plot_burn_attitude(ax, data):
    sat = sat_sim.SatSimBurn()
    for i, r in enumerate(data):
        ang_err = []
        q_a = r['xf'][sat.model['x_index']['q0_a']:sat.model['x_index']['q3_a']+1, :]
        q_b = r['xf'][sat.model['x_index']['q0_b']:sat.model['x_index']['q3_b']+1, :]
        t = r['xf'][0, :].full().squeeze()
        for j in range(len(t)):
            err = ca.DM(f_ang_err(q_a[:, j], q_b[:, j])).full().squeeze()
            ang_err.append(err)
        ang_err = np.array(ang_err).T
        h_roll = ax.plot(t, np.rad2deg(ang_err[0, :]), 'r', alpha=0.2)
        h_pitch = ax.plot(t, np.rad2deg(ang_err[1, :]), 'g', alpha=0.2)
        h_yaw = ax.plot(t, np.rad2deg(ang_err[2, :]), 'b', alpha=0.2)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Att. Error [deg]')

    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='roll', alpha=0.5),
        Line2D([0], [0], color='green', lw=2, label='pitch', alpha=0.5),
        Line2D([0], [0], color='blue', lw=2, label='yaw', alpha=0.5 ),
    ]
    ax.legend(handles=legend_elements, ncol=3, loc='best')

    ax.grid()
    return ax

def plot_burn_delta_vx_error(ax, data):
    sat = sat_sim.SatSimBurn()
    for i, r in enumerate(data):
        ang_err = []
        v_a = r['xf'][sat.model['x_index']['vx_a'], :]
        v_b = r['xf'][sat.model['x_index']['vx_b'], :]
        v_err = ca.DM(v_a - v_b).full().squeeze()
        t = r['xf'][0, :].full().squeeze()
        h_x = ax.plot(t, v_err, 'b', alpha=0.2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Vx Error [m/s]')
    ax.grid()
    return ax

def plot_burn_delta_vy_error(ax, data):
    sat = sat_sim.SatSimBurn()
    for i, r in enumerate(data):
        ang_err = []
        v_a = r['xf'][sat.model['x_index']['vy_a'], :]
        v_b = r['xf'][sat.model['x_index']['vy_b'], :]
        v_err = ca.DM(v_a - v_b).full().squeeze()
        t = r['xf'][0, :].full().squeeze()
        h_x = ax.plot(t, v_err, 'b', alpha=0.2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Vy Error [m/s]')
    ax.grid()
    return ax

def plot_burn_delta_vz_error(ax, data):
    sat = sat_sim.SatSimBurn()
    for i, r in enumerate(data):
        ang_err = []
        v_a = r['xf'][sat.model['x_index']['vz_a'], :]
        v_b = r['xf'][sat.model['x_index']['vz_b'], :]
        v_err = ca.DM(v_a - v_b).full().squeeze()
        t = r['xf'][0, :].full().squeeze()
        h_x = ax.plot(t, v_err, 'b', alpha=0.2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Vz Error [m/s]')
    ax.grid()
    return ax
    

def plot_orbits(ax, data_ref, data):
    sat = sat_sim.SatSimCoast()
    ax.set_title('Monte Carlo Simulation of Spacecraft with Disturbances')
    xind = sat.model['x_index']

    px_b = data_ref['xf'][xind['px_a'], :].full().squeeze()
    py_b = data_ref['xf'][xind['py_a'], :].full().squeeze()
    vx_b = data_ref['xf'][xind['vx_a'], :].full().squeeze()
    vy_b = data_ref['xf'][xind['vy_a'], :].full().squeeze()
    
    for i, r in enumerate(data):
        px_a = r['xf'][xind['px_a'], :].full().squeeze()
        py_a = r['xf'][xind['py_a'], :].full().squeeze()
        h_traj = ax.plot(px_a, py_a, linewidth=0.5, color='#D62728', alpha=0.4, zorder=1)

    # plot nominal trajectory with white halo to make it more visible
    ax.plot(px_b, py_b, linewidth=4, color='white', alpha=0.4, zorder=2)
    h_ref = ax.plot(px_b, py_b, linewidth=3, color='#008B8B', alpha=1, zorder=3)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    h_earth = ax.add_patch(Circle((0, 0), 6371e3, linewidth=0.5, facecolor='#1F77B4', edgecolor='black', fill=True, label='Earth'))

    h_sc = ax.scatter(px_b[0], py_b[0], color='#FFD700', linewidth=0.5, edgecolor='black', s=50, zorder=4)

    scale_factor = 1e4   # adjust for visual clarity
    h_quiv = ax.quiver(
        px_b[0], py_b[0], vx_b[0] * scale_factor, vy_b[0]*scale_factor, angles='xy', scale_units='xy',
        scale=1, color='#228B22', width=0.01, zorder=2)

    arrow_handle = FancyArrowPatch((0, 0), (1, 0), color='#228B22', lw=2, label='ΔV Direction')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1F77B4',
            markeredgecolor='black', markeredgewidth=0.5, lw=0.5, markersize=10, label='Earth'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700',
            markeredgecolor='black', markeredgewidth=0.5,lw=0.5, markersize=10, label='Spacecraft Start'),
        Line2D([0], [0], color='#008B8B', lw=3, alpha=1, label='Nominal Orbit'),
        Line2D([0], [0], color='#D62728', lw=2, alpha=0.5, label='Disturbed Orbits'),
        arrow_handle
    ]

    ax.legend(handler_map={FancyArrowPatch: HandlerArrow()}, handles=legend_elements, ncol=1, loc='best', frameon=True)

    ax.grid()
    ax.axis('equal')
    return ax

def plot_quaternion_norm_error(ax, sat, data):
    q_a = data[0]['xf'][sat.model['x_index']['q0_a']:sat.model['x_index']['q3_a']+1, :]
    q_b = data[0]['xf'][sat.model['x_index']['q0_b']:sat.model['x_index']['q3_b']+1, :]

    ax.plot(np.linalg.norm(q_a, axis=0)[:2000] - 1, label='state')
    ax.plot(np.linalg.norm(q_b, axis=0)[:2000] - 1, label='reference')
    ax.set_title('Quaternion Norm Error During Burn Phase')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Quaternion Norm Error')
    ax.grid()
    ax.legend(loc='best')
    return ax