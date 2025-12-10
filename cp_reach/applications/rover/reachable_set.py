# Copyright 2025 Micah Condie
import math
import numpy as np
import sympy
import sympy.physics.mechanics as me
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

def emi_disturbance(rover, ax):
    # Plot Abstract Analysis
    pi= math.pi
    disturbance = rover['emi_disturbance'] # Magnetometer Disturbance.
    heading = rover['heading']
    turn_radius = rover['turn_radius']

    # 1 - Positive disturbance
    # 2 - Negative disturbance     

    # Initial direction
    psi_0 = heading + pi/2

    # Phase 1 (go straight 7m)
    x_1 = 7*math.cos(psi_0)
    y_1 = 7*math.sin(psi_0) 

    X1 = [0, x_1]
    Y1 = [0, y_1]
    ax.plot(X1,Y1, color="blue")
    
    # Phase 2 (turn)
    # Locate center point of arc, based on turn radius.
    gamma = (90-psi_0*180/pi)*pi/180
    center_point = [x_1 + turn_radius*math.cos(gamma), y_1-turn_radius*math.sin(gamma)]

    psi1_2 = psi_0 + (disturbance)*pi/180
    psi2_2 = psi_0 - (disturbance)*pi/180

    # First trajectory leaves early.
    psi1 = np.linspace(psi_0 + 90*pi/180, psi1_2, 100)
    # Second trajectory leaves late.
    psi2 = np.linspace(psi1_2, psi2_2, 100)
    
    # First and second trajectory
    X1 = turn_radius*np.cos(psi1) + center_point[0]
    Y1 = turn_radius*np.sin(psi1) + center_point[1]
    X2 = turn_radius*np.cos(psi2) + center_point[0]
    Y2 = turn_radius*np.sin(psi2) + center_point[1]

    ax.plot(X1,Y1, color="blue")
    ax.plot(X2,Y2, color="blue")
    

    # Obtain terminal points.
    X_end_bound = []
    Y_end_bound = []
    for idx in range(len(psi2)):
        if psi2[idx] < psi1_2:
            X_end_bound.append(7*np.sin(psi2[idx]) + X2[idx])
            Y_end_bound.append(-7*np.cos(psi2[idx]) + Y2[idx])

    ax.plot(X_end_bound, Y_end_bound, color="blue")

    
    # End of phase 2
    x1_2 = X1[-1]
    y1_2 = Y1[-1]
    x2_2 = X2[-1]
    y2_2 = Y2[-1]

    # min and max end points
    x1_end = X_end_bound[0]
    x2_end = X_end_bound[-1]
    y1_end = Y_end_bound[0]
    y2_end = Y_end_bound[-1]

    X1_leg = [x1_2, x1_end]
    Y1_leg = [y1_2, y1_end]
    X2_leg = [x2_2, x2_end]
    Y2_leg = [y2_2, y2_end]

    ax.plot(X1_leg,Y1_leg, color="blue")
    ax.plot(X2_leg,Y2_leg, color="blue", label= "Rover Trajectories")



    points2 = list(zip(X2[::-1], Y2[::-1]))
    end_points = list(zip(X_end_bound[::-1], Y_end_bound[::-1]))


    # end_points1 = list(zip(X_end_bound[:1], Y_end_bound[:1]))
    # polygon1_points =  points2 + end_points1
    # poly1 = Polygon(polygon1_points, closed=True, facecolor='lightblue', edgecolor='none', alpha=0.5, zorder=0)
    # ax.add_patch(poly1)


    points2_terminal = list(zip(X2[:-2], Y2[:-2]))
    polygon1_points =  points2_terminal + end_points
    poly2 = Polygon(polygon1_points, closed=True, facecolor='lightblue', edgecolor='none', alpha=0.5, label="Reachable Set", zorder=0)
    ax.add_patch(poly2)
    
  
    ax.scatter(0, 0, color="black",zorder=5)
    ax.annotate('Rover Starting Position', (4.5, -0.6), textcoords="offset points", xytext=(10,10), ha='center', zorder=10)
    # ax.scatter(x_f, y_f, color="black",zorder=5)
    # ax.annotate('Goal', (x_f-1, y_f-0.6), textcoords="offset points", xytext=(10,10), ha='center', zorder=10)
    # ax = ax.subplot(1,1,1)
    

    ax.set_xlabel('Position of Rover (m)')
    ax.set_ylabel('Position of Rover (m)')
    ax.set_axisbelow(True)
    ax.grid(True)
    # # ax.legend(loc=1)
    ax.axis([-5, 25, -5, 25])
    # ax.axis('equal')
    #ax.tight_layout()
    ax.set_title(f'Reachable set for Rover With a Disturbance of {disturbance:.0f}$\\degree$', fontsize=14)
    ax.legend()
    # plt.savefig("fig/rover_reachable_positions.png")
    # plt.savefig("/home/micah/example/rover_reachable_positions.png")
    
    # plt.close()



def roll_over(rover, ax):
    m, g, v, r = sympy.symbols('m, g, v, r', real=True)
    theta_t, theta_f, theta, l = sympy.symbols('theta_t, theta_f, theta, l',real=True)
    frame_e = me.ReferenceFrame('e') # earth local level frame
    frame_t = frame_e.orientnew('t', 'Axis', (frame_e.z, theta_t)) # terrain frame
    frame_b = frame_t.orientnew('b', 'Axis', (frame_e.z, theta)) # terrain frame
    frame_f = frame_b.orientnew('f', 'Axis', (frame_e.z, theta_f)) # body frame
    # position vector from tire rotation point to center of mass
    r_ap = l*frame_f.x
    W = -m*g*frame_e.y
    Fc = -(m*v**2/r)*frame_t.x
    F = W + Fc
    M = r_ap.cross(F)

    eq1 = (M.to_matrix(frame_e)[2]/(g*l*m)).simplify()
    eq2 = eq1.subs(theta, 0)
    v_expr = sympy.solve(eq1.subs(theta, 0), v)[1]

    def plot_roll_over_analysis(theta_f_val,rad):
        print(rad)
        f_v_expr = sympy.lambdify([g, r, theta_f, theta_t], [v_expr])
        for r_val in [rad]:
            theta_t_vals = np.linspace(0, np.pi/2-theta_f_val, 1000)
            v_vals = f_v_expr(g=9.8, r=r_val, theta_f=theta_f_val,theta_t=theta_t_vals)[0]
            ax.plot(np.rad2deg(theta_t_vals), v_vals, label='r={:4.0f} m'.format(r_val))
            # print(f'Rollover Velocity for 0 Terrain Angle: {v_vals[0]}')

        ax.fill_between(np.rad2deg(theta_t_vals), v_vals, 0, alpha=0.3, color="green",label='safe')
        ax.fill_between(np.rad2deg(theta_t_vals), v_vals, 50, alpha=0.3, color="red",label='unsafe')

        ax.grid()
        ax.set_xlabel('Terrain Angle [deg]')
        ax.set_ylabel('Velocity for Roll Over [m/s]')
        ax.set_title('Roll Over Velocity $\\theta_f$ = {:0.1f} deg'.format(np.rad2deg(theta_f_val)))
        ax.legend()
        ax.set_ylim(bottom=0,top=17)

    #lx = 1/2 rover width
    lx = rover['width']/2#0.105
    #ly = COM to ground
    ly = rover['COM_height']#0.06
    # Turn radius
    rad=rover['turn_radius']
    plot_roll_over_analysis(theta_f_val=np.arctan(ly/lx), rad=rad)