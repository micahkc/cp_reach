// =====================================================================
// 6-DOF Rigid Body Kinematics (rotation matrix representation)
//
// States: position p (world/NED), velocity v (world/NED), rotation R (body-to-world)
// Free variables (to be defined by extending model):
//   a_x, a_y, a_z          body-frame specific force (excluding gravity)
//   omega_x, omega_y, omega_z   body-frame angular velocity
//
// Equations:
//   p_dot = v
//   v_dot = R * a + g
//   R_dot = R * [omega]_x
// =====================================================================

class RigidBody

  parameter Real g = 9.80665 "Gravity [m/s^2]";

  // Position in world frame (NED)
  Real px(start = 0) "Position North [m]";
  Real py(start = 0) "Position East [m]";
  Real pz(start = 0) "Position Down [m]";

  // Velocity in world frame (NED)
  Real vx(start = 0) "Velocity North [m/s]";
  Real vy(start = 0) "Velocity East [m/s]";
  Real vz(start = 0) "Velocity Down [m/s]";

  // Rotation matrix R (body-to-world), initialized to identity
  Real R11(start = 1), R12(start = 0), R13(start = 0);
  Real R21(start = 0), R22(start = 1), R23(start = 0);
  Real R31(start = 0), R32(start = 0), R33(start = 1);

  // Body-frame angular velocity
  Real omega_x "Body roll rate [rad/s]";
  Real omega_y "Body pitch rate [rad/s]";
  Real omega_z "Body yaw rate [rad/s]";

  // Body-frame specific force (excluding gravity)
  Real a_x "Body specific force X [m/s^2]";
  Real a_y "Body specific force Y [m/s^2]";
  Real a_z "Body specific force Z [m/s^2]";

equation

  // p_dot = v
  der(px) = vx;
  der(py) = vy;
  der(pz) = vz;

  // v_dot = R * a + g_vec, where g_vec = [0, 0, g] in NED
  der(vx) = R11 * a_x + R12 * a_y + R13 * a_z;
  der(vy) = R21 * a_x + R22 * a_y + R23 * a_z;
  der(vz) = R31 * a_x + R32 * a_y + R33 * a_z + g;

  // R_dot = R * [omega]_x
  der(R11) =  R12 * omega_z - R13 * omega_y;
  der(R12) =  R13 * omega_x - R11 * omega_z;
  der(R13) =  R11 * omega_y - R12 * omega_x;
  der(R21) =  R22 * omega_z - R23 * omega_y;
  der(R22) =  R23 * omega_x - R21 * omega_z;
  der(R23) =  R21 * omega_y - R22 * omega_x;
  der(R31) =  R32 * omega_z - R33 * omega_y;
  der(R32) =  R33 * omega_x - R31 * omega_z;
  der(R33) =  R31 * omega_y - R32 * omega_x;

end RigidBody;


// =====================================================================
// Quadrotor plant (extends RigidBody with angular dynamics)
//
// Adds: Euler's equations for angular rates, thrust-to-force mapping
// States (added): omega_x, omega_y, omega_z
// Free variables (to be defined by extending model):
//   T, Mx, My, Mz    thrust and body-frame moments
// =====================================================================

model Quadrotor

  extends RigidBody;

  parameter Real mass = 2.496 "Total mass [kg]";
  parameter Real Ixx = 0.0233 "Roll moment of inertia [kg*m^2]";
  parameter Real Iyy = 0.0344 "Pitch moment of inertia [kg*m^2]";
  parameter Real Izz = 0.0527 "Yaw moment of inertia [kg*m^2]";

  // Thrust and moments (to be defined by controller)
  Real T "Total thrust [N]";
  Real Mx "Roll moment [N*m]";
  Real My "Pitch moment [N*m]";
  Real Mz "Yaw moment [N*m]";

equation

  // Body-frame specific force: thrust along body -Z (NED: down is positive)
  a_x = 0;
  a_y = 0;
  a_z = -T / mass;

  // Angular rate dynamics (Euler's equations, diagonal inertia)
  der(omega_x) = (Mx + (Iyy - Izz) * omega_y * omega_z) / Ixx;
  der(omega_y) = (My + (Izz - Ixx) * omega_x * omega_z) / Iyy;
  der(omega_z) = (Mz + (Ixx - Iyy) * omega_x * omega_y) / Izz;

end Quadrotor;


// =====================================================================
// Closed-loop quadrotor with geometric tracking controller + feedforward
//
// Geometric controller (Lee, Leok, McClamroch 2010):
//   Position feedback changes the desired force direction, which changes
//   R_des. The attitude controller tracks R_des (not just R_ref).
//   This closes the position -> attitude -> thrust direction loop,
//   consistent with the SE_2(3) reachability analysis.
//
// Control structure:
//   u_total = u_ff + u_fb
//
// Outer loop (translation):
//   a_fb    = -Kp*e_p - Kv*e_v - Ki*integral(e_v)
//   F_des   = mass * (a_total - g_vec)
//   T       = -F_des . (R * e3)
//   R_des   = rotation aligning body z with -F_des / |F_des|
//
// Inner loop (rotation):
//   e_R  = 0.5 * vee(R_des^T R - R^T R_des)
//   e_w  = (omega + d) - omega_ref
//   M    = M_ff - I*(Kr*e_R + Kw*e_w) + omega x I*omega
//
// States (from Quadrotor): px,py,pz, vx,vy,vz, R[9], omega[3] = 18
// States (added): vel_err_ix, vel_err_iy, vel_err_iz = 3  (total: 21)
// =====================================================================

model QuadrotorClosedLoop2

  extends Quadrotor;

  // -----------------------------------------------------------------
  // Controller gains
  // -----------------------------------------------------------------
  // Position → velocity
  parameter Real Kp_px = 1.0 "Position X P gain [1/s]";
  parameter Real Kp_py = 1.0 "Position Y P gain [1/s]";
  parameter Real Kp_pz = 1.5 "Position Z P gain [1/s]";

  // Velocity → desired acceleration (PI)
  parameter Real Kp_vx = 2.0 "Velocity X P gain [1/s]";
  parameter Real Ki_vx = 0.1 "Velocity X I gain [1/s^2]";
  parameter Real Kp_vy = 2.0 "Velocity Y P gain [1/s]";
  parameter Real Ki_vy = 0.1 "Velocity Y I gain [1/s^2]";
  parameter Real Kp_vz = 2.5 "Velocity Z P gain [1/s]";
  parameter Real Ki_vz = 0.1 "Velocity Z I gain [1/s^2]";

  // Attitude → rate (SO(3) geometric)
  parameter Real Kr_x = 6.0 "Attitude X gain [1/s]";
  parameter Real Kr_y = 6.0 "Attitude Y gain [1/s]";
  parameter Real Kr_z = 4.0 "Attitude Z gain [1/s]";

  // Rate → moment (PD)
  parameter Real Kw_x = 5.0 "Rate X P gain";
  parameter Real Kw_y = 5.0 "Rate Y P gain";
  parameter Real Kw_z = 4.0 "Rate Z P gain";

  // -----------------------------------------------------------------
  // Reference inputs
  // -----------------------------------------------------------------
  // Position and velocity reference (world frame)
  input Real px_ref = 0.0 "Reference position X [m]";
  input Real py_ref = 0.0 "Reference position Y [m]";
  input Real pz_ref = 0.0 "Reference position Z [m]";
  input Real vx_ref = 0.0 "Reference velocity X [m/s]";
  input Real vy_ref = 0.0 "Reference velocity Y [m/s]";
  input Real vz_ref = 0.0 "Reference velocity Z [m/s]";

  // Acceleration feedforward (world frame, from differential flatness)
  input Real ax_ref = 0.0 "Reference acceleration X [m/s^2]";
  input Real ay_ref = 0.0 "Reference acceleration Y [m/s^2]";
  input Real az_ref = 0.0 "Reference acceleration Z [m/s^2]";

  // Rotation matrix reference (body-to-world), for hover: identity
  // Used for yaw direction (b1_c) and feedforward angular rate/moments
  input Real R11_ref(start = 1) = 1.0 "Reference R(1,1)";
  input Real R12_ref(start = 0) = 0.0 "Reference R(1,2)";
  input Real R13_ref(start = 0) = 0.0 "Reference R(1,3)";
  input Real R21_ref(start = 0) = 0.0 "Reference R(2,1)";
  input Real R22_ref(start = 1) = 1.0 "Reference R(2,2)";
  input Real R23_ref(start = 0) = 0.0 "Reference R(2,3)";
  input Real R31_ref(start = 0) = 0.0 "Reference R(3,1)";
  input Real R32_ref(start = 0) = 0.0 "Reference R(3,2)";
  input Real R33_ref(start = 1) = 1.0 "Reference R(3,3)";

  // Angular rate reference (body frame)
  input Real omega_x_ref = 0.0 "Reference roll rate [rad/s]";
  input Real omega_y_ref = 0.0 "Reference pitch rate [rad/s]";
  input Real omega_z_ref = 0.0 "Reference yaw rate [rad/s]";

  // Feedforward moments (from inverse dynamics / flatness)
  input Real Mx_ff = 0.0 "Feedforward roll moment [N*m]";
  input Real My_ff = 0.0 "Feedforward pitch moment [N*m]";
  input Real Mz_ff = 0.0 "Feedforward yaw moment [N*m]";

  // -----------------------------------------------------------------
  // Disturbance inputs (acoustic attack injection on gyro measurement)
  // -----------------------------------------------------------------
  input Real d_p = 0.0 "Disturbance on roll rate measurement [rad/s]";
  input Real d_q = 0.0 "Disturbance on pitch rate measurement [rad/s]";
  input Real d_r = 0.0 "Disturbance on yaw rate measurement [rad/s]";

  // -----------------------------------------------------------------
  // Integrator states (velocity error integral for PI control)
  // -----------------------------------------------------------------
  Real vel_err_ix(start = 0) "Velocity error integral X [m]";
  Real vel_err_iy(start = 0) "Velocity error integral Y [m]";
  Real vel_err_iz(start = 0) "Velocity error integral Z [m]";

  // -----------------------------------------------------------------
  // Algebraic: errors and controller intermediates
  // -----------------------------------------------------------------
  // Position error (world frame)
  Real e_px "Position error X [m]";
  Real e_py "Position error Y [m]";
  Real e_pz "Position error Z [m]";

  // Velocity error (world frame)
  Real e_vx "Velocity error X [m/s]";
  Real e_vy "Velocity error Y [m/s]";
  Real e_vz "Velocity error Z [m/s]";

  // Feedback acceleration from outer loop (world frame)
  Real a_fb_x "Feedback acceleration X [m/s^2]";
  Real a_fb_y "Feedback acceleration Y [m/s^2]";
  Real a_fb_z "Feedback acceleration Z [m/s^2]";

  // Total desired acceleration = feedforward + feedback (world frame)
  Real a_total_x "Total desired acceleration X [m/s^2]";
  Real a_total_y "Total desired acceleration Y [m/s^2]";
  Real a_total_z "Total desired acceleration Z [m/s^2]";

  // Desired force (world frame)
  Real F_des_x "Desired force X [N]";
  Real F_des_y "Desired force Y [N]";
  Real F_des_z "Desired force Z [N]";
  Real F_norm "Desired force magnitude [N]";

  // Desired rotation R_des computed from F_des direction + yaw from R_ref
  // b3_des = -F_des / |F_des| (desired body z, aligns thrust with -F_des)
  Real b3_x, b3_y, b3_z;
  // b1_c = first column of R_ref (desired heading direction from flatness)
  // b2_des = normalize(b3_des x b1_c)
  Real b2_unnorm_x, b2_unnorm_y, b2_unnorm_z, b2_norm;
  Real b2_x, b2_y, b2_z;
  // b1_des = b2_des x b3_des
  Real b1_x, b1_y, b1_z;

  // Attitude error via vee map: e_R = 0.5 * vee(R_des^T * R - R^T * R_des)
  Real e_Rx "Attitude error about X (roll) [rad]";
  Real e_Ry "Attitude error about Y (pitch) [rad]";
  Real e_Rz "Attitude error about Z (yaw) [rad]";

  // Elements of R_des^T * R (for vee map computation)
  Real S11, S12, S13;
  Real S21, S22, S23;
  Real S31, S32, S33;

  // Angular rate error (with disturbance on measurement)
  Real e_wx "Rate error X [rad/s]";
  Real e_wy "Rate error Y [rad/s]";
  Real e_wz "Rate error Z [rad/s]";

equation

  // =================================================================
  // Position error
  // =================================================================
  e_px = px - px_ref;
  e_py = py - py_ref;
  e_pz = pz - pz_ref;

  // =================================================================
  // Velocity error
  // =================================================================
  e_vx = vx - vx_ref;
  e_vy = vy - vy_ref;
  e_vz = vz - vz_ref;

  // Velocity error integrators
  der(vel_err_ix) = e_vx;
  der(vel_err_iy) = e_vy;
  der(vel_err_iz) = e_vz;

  // =================================================================
  // Outer loop: feedback acceleration (PID on position/velocity error)
  // =================================================================
  a_fb_x = -Kp_px*e_px - Kp_vx*e_vx - Ki_vx*vel_err_ix;
  a_fb_y = -Kp_py*e_py - Kp_vy*e_vy - Ki_vy*vel_err_iy;
  a_fb_z = -Kp_pz*e_pz - Kp_vz*e_vz - Ki_vz*vel_err_iz;

  // =================================================================
  // Total desired acceleration = feedforward + feedback
  // =================================================================
  a_total_x = ax_ref + a_fb_x;
  a_total_y = ay_ref + a_fb_y;
  a_total_z = az_ref + a_fb_z;

  // =================================================================
  // Desired force and R_des (geometric controller)
  //
  //   F_des = mass * (a_total - g_vec), g_vec = [0, 0, g] in NED
  //   b3_des = -F_des / |F_des|  (body z opposes desired force)
  //   b1_c = R_ref * e1           (yaw direction from flatness)
  //   b2_des = normalize(b3_des x b1_c)
  //   b1_des = b2_des x b3_des
  //   R_des = [b1_des | b2_des | b3_des]
  //
  // At hover (a_total=0): F_des = [0, 0, -mg], b3_des = [0,0,1],
  //   R_des = I. Correct.
  // =================================================================
  F_des_x = mass * a_total_x;
  F_des_y = mass * a_total_y;
  F_des_z = mass * (a_total_z - g);
  F_norm = sqrt(F_des_x*F_des_x + F_des_y*F_des_y + F_des_z*F_des_z);

  // b3_des: desired body z-axis (points opposite to desired force)
  b3_x = -F_des_x / F_norm;
  b3_y = -F_des_y / F_norm;
  b3_z = -F_des_z / F_norm;

  // b2_des = normalize(b3_des x b1_c), where b1_c = R_ref first column
  b2_unnorm_x = b3_y * R31_ref - b3_z * R21_ref;
  b2_unnorm_y = b3_z * R11_ref - b3_x * R31_ref;
  b2_unnorm_z = b3_x * R21_ref - b3_y * R11_ref;
  b2_norm = sqrt(b2_unnorm_x*b2_unnorm_x + b2_unnorm_y*b2_unnorm_y + b2_unnorm_z*b2_unnorm_z);
  b2_x = b2_unnorm_x / b2_norm;
  b2_y = b2_unnorm_y / b2_norm;
  b2_z = b2_unnorm_z / b2_norm;

  // b1_des = b2_des x b3_des
  b1_x = b2_y * b3_z - b2_z * b3_y;
  b1_y = b2_z * b3_x - b2_x * b3_z;
  b1_z = b2_x * b3_y - b2_y * b3_x;

  // =================================================================
  // Thrust: project desired force onto body Z axis
  //   T = -F_des . (R * e3), where R*e3 = [R13, R23, R33]
  // =================================================================
  T = -(F_des_x * R13 + F_des_y * R23 + F_des_z * R33);

  // =================================================================
  // Attitude error: geometric SO(3) error via vee map
  //
  //   S = R_des^T * R  (relative rotation from R_des to actual R)
  //   e_R = 0.5 * vee(S - S^T)
  //
  // R_des = [b1 | b2 | b3], so R_des^T has rows b1^T, b2^T, b3^T.
  // =================================================================

  // Compute S = R_des^T * R
  S11 = b1_x*R11 + b1_y*R21 + b1_z*R31;
  S12 = b1_x*R12 + b1_y*R22 + b1_z*R32;
  S13 = b1_x*R13 + b1_y*R23 + b1_z*R33;
  S21 = b2_x*R11 + b2_y*R21 + b2_z*R31;
  S22 = b2_x*R12 + b2_y*R22 + b2_z*R32;
  S23 = b2_x*R13 + b2_y*R23 + b2_z*R33;
  S31 = b3_x*R11 + b3_y*R21 + b3_z*R31;
  S32 = b3_x*R12 + b3_y*R22 + b3_z*R32;
  S33 = b3_x*R13 + b3_y*R23 + b3_z*R33;

  // Vee map: extract rotation error from skew-symmetric part of S
  // vee([0, -c, b; c, 0, -a; -b, a, 0]) = [a, b, c]
  e_Rx = 0.5 * (S32 - S23);
  e_Ry = 0.5 * (S13 - S31);
  e_Rz = 0.5 * (S21 - S12);

  // =================================================================
  // Angular rate error (disturbance injected on measurement)
  // Controller sees (omega + d) instead of omega
  // =================================================================
  e_wx = (omega_x + d_p) - omega_x_ref;
  e_wy = (omega_y + d_q) - omega_y_ref;
  e_wz = (omega_z + d_r) - omega_z_ref;

  // =================================================================
  // Moments: feedforward + geometric attitude + rate PD + gyroscopic
  //
  // Gyroscopic compensation: cancels the -(omega x J*omega) coupling
  // in Euler's equations.
  // =================================================================
  Mx = Mx_ff - Ixx*(Kr_x*e_Rx + Kw_x*e_wx) + (Izz - Iyy)*omega_y*omega_z;
  My = My_ff - Iyy*(Kr_y*e_Ry + Kw_y*e_wy) + (Ixx - Izz)*omega_x*omega_z;
  Mz = Mz_ff - Izz*(Kr_z*e_Rz + Kw_z*e_wz) + (Iyy - Ixx)*omega_x*omega_y;

end QuadrotorClosedLoop2;
