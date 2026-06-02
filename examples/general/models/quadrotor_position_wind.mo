// Position tracking via Log-Linear Dynamic Inversion (LLDI) on SE_2(3).
// Matches Lin (2026) dissertation eq 4.15 / 4.17 / Fig 4.1 architecture.
//
// Top-level inputs (4):
//   px_ref, py_ref, pz_ref  - NWU world position reference [m]
//   wind_speed              - wind disturbance [m/s] along wind_dir_world
//
// Controller pipeline:
//   1. Tracking differentiators: position -> (v_ref, a_ref, j_ref) per axis
//      (high-gain observer, 4 states per axis, all poles at -omega_bw)
//   2. Flatness: a_ref_world + yaw=0 -> R_ref ; j_ref -> omega_ref_body
//   3. Build X (SE_2(3) state) from plant feedback; build X_ref likewise
//   4. Lie algebra error ζ = log_SE_2(3)(X^-1 X_ref), 9-dim, body frame
//      Small-angle approximation (J_l^-1 ≈ I_3) for ζ_p, ζ_v
//   5. LQR state feedback: u_ζ = -K · ζ ∈ R^4 = (δa_z, δω_x, δω_y, δω_z)
//   6. Add feedforward: a_z_cmd = a_z_ff + u_ζ[1] ; ω_cmd = ω_ff + u_ζ[2:4]
//   7. Inner rate loop (eq 4.17 simplified, no snap FF):
//      M_b = J·K_ω·(ω_cmd - ω_meas) + ω × J·ω
//   8. Motor mixing -> omega_cmd[4]
//
// Frame conventions: world NWU (z up), body FLU. Plant exposes gyro in FRD,
// so omega_b_FLU = {gyro[1], -gyro[2], -gyro[3]}.
//
// The K matrix (4×9 LQR gain) is the load-bearing parameter for matching
// cp_reach's flowpipe analysis. Compute via control.lqr(A_eq, B_u, Q, R)
// at the hover linearization (or trajectory-averaged ν̄_vr) and paste below.
// The default K is a hand-picked nested-PID-equivalent that flies but is
// NOT LQR-optimal — recompute for tight cp_reach bound matching.


model TrackingDifferentiator
  // 4-state per-axis high-gain observer. All poles placed at -omega_bw.
  // Char poly: (s + ω_bw)^4 = s^4 + 4ω·s^3 + 6ω²·s^2 + 4ω³·s + ω^4
  parameter Real omega_bw = 25.0 "Bandwidth [rad/s]";
  parameter Real k1 = 4.0 * omega_bw;
  parameter Real k2 = 6.0 * omega_bw^2;
  parameter Real k3 = 4.0 * omega_bw^3;
  parameter Real k4 = omega_bw^4;

  input Real x_in;

  output Real p_hat(start = 0, fixed = true);
  output Real v_hat(start = 0, fixed = true);
  output Real a_hat(start = 0, fixed = true);
  output Real j_hat(start = 0, fixed = true);

protected
  Real err;

equation
  err = x_in - p_hat;
  der(p_hat) = v_hat + k1 * err;
  der(v_hat) = a_hat + k2 * err;
  der(a_hat) = j_hat + k3 * err;
  der(j_hat) = k4 * err;
end TrackingDifferentiator;


model LLDIController
  // Plant parameters (must match QuadrotorSIL)
  parameter Real mass = 2.0;
  parameter Real g = 9.8 "Gravity magnitude [m/s^2]";
  parameter Real Ct = 8.54858e-6;
  parameter Real Cm = 0.016;
  parameter Real arm_length = 0.25;
  parameter Real d = arm_length * 0.7071067811865476;
  parameter Real omega_cmd_max = 1600.0;
  parameter Real Ixx = 0.02166666666666667;
  parameter Real Iyy = 0.02166666666666667;
  parameter Real Izz = 0.04;

  // LQR gain K (4x9), acts on ζ = [ζ_p(3); ζ_v(3); ζ_R(3)] in body frame.
  // u_ζ = -K · ζ = (δa_z, δω_x, δω_y, δω_z).
  // Default below is a hand-tuned nested controller; replace with the
  // K from cp_reach (control.lqr) for cp_reach bound matching.
  parameter Real K[4, 9] = [
    //  ζ_p_x  ζ_p_y  ζ_p_z   ζ_v_x  ζ_v_y  ζ_v_z   ζ_R_x  ζ_R_y  ζ_R_z
        0,     0,    -2.5,    0,     0,    -2.5,    0,     0,     0;     // δa_z
        0,     1.2,   0,      0,     1.8,   0,     -8.0,   0,     0;     // δω_x  (roll)
       -1.2,   0,     0,     -1.8,   0,     0,      0,    -8.0,   0;     // δω_y  (pitch)
        0,     0,     0,      0,     0,     0,      0,     0,    -3.0    // δω_z  (yaw)
  ];

  // Inner rate-loop gains (eq 4.17 — body frame P controller)
  parameter Real K_omega[3] = {20.0, 20.0, 10.0} "Rate loop P gains [1/s]";

  // Reference position derivatives (from TD)
  input Real p_ref[3];
  input Real v_ref[3];
  input Real a_ref[3];
  input Real j_ref[3];

  // Plant feedback
  input Real position[3](start = {0, 0, 0});
  input Real velocity[3](start = {0, 0, 0});
  input Real quat[4](start = {1, 0, 0, 0});
  input Real gyro[3](start = {0, 0, 0});

  output Real omega_cmd[4];

protected
  // Plant state in body-to-world rotation
  Real R[3, 3];
  Real omega_b[3] "Body angular velocity in FLU [rad/s]";

  // Reference attitude from flatness
  Real f_world[3] "Specific force vector in world (NWU) [N]";
  Real T_ff "Feed-forward total thrust [N]";
  Real a_z_ff "Feed-forward body-z thrust accel [m/s^2]";
  Real z_b_ref[3];
  Real x_c[3];
  Real y_b_ref_raw[3];
  Real y_b_ref_norm;
  Real y_b_ref[3];
  Real x_b_ref[3];
  Real R_ref[3, 3];

  // Reference body rates from jerk (NWU sign convention)
  Real t_scaled[3];
  Real omega_ff_b[3];

  // SE_2(3) Lie algebra error in body frame
  Real S[3, 3] "R^T · R_ref";
  Real trace_S;
  Real arg, arg_safe;
  Real theta, sin_theta;
  Real log_factor;
  Real zeta_R[3], zeta_v[3], zeta_p[3];
  Real zeta[9];

  // LQR control law output (4-dim)
  Real u_zeta[4];

  // Total control commands
  Real a_z_cmd;
  Real T_cmd;
  Real omega_cmd_b[3];

  // Inner rate loop
  Real omega_err[3];
  Real omega_dot_cmd[3];
  Real Jw[3];
  Real M_b[3];

  // Motor mixing
  Real F_motor[4];
  Real F_motor_safe[4];
  Real F_max;

equation
  // Quaternion (w,x,y,z) -> body-to-world rotation matrix
  R[1, 1] = 1 - 2*(quat[3]*quat[3] + quat[4]*quat[4]);
  R[1, 2] = 2*(quat[2]*quat[3] - quat[1]*quat[4]);
  R[1, 3] = 2*(quat[2]*quat[4] + quat[1]*quat[3]);
  R[2, 1] = 2*(quat[2]*quat[3] + quat[1]*quat[4]);
  R[2, 2] = 1 - 2*(quat[2]*quat[2] + quat[4]*quat[4]);
  R[2, 3] = 2*(quat[3]*quat[4] - quat[1]*quat[2]);
  R[3, 1] = 2*(quat[2]*quat[4] - quat[1]*quat[3]);
  R[3, 2] = 2*(quat[3]*quat[4] + quat[1]*quat[2]);
  R[3, 3] = 1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]);

  // FRD gyro -> FLU body rate
  omega_b[1] = gyro[1];
  omega_b[2] = -gyro[2];
  omega_b[3] = -gyro[3];

  // ===== Reference attitude from flatness (yaw=0, NWU) =====
  // f = m * (a_ref + g*ẑ_world). Body z aligns with f.
  f_world[1] = mass * a_ref[1];
  f_world[2] = mass * a_ref[2];
  f_world[3] = mass * (a_ref[3] + g);
  T_ff = sqrt(f_world[1]^2 + f_world[2]^2 + f_world[3]^2 + 1e-9);
  a_z_ff = T_ff / mass;

  z_b_ref[1] = f_world[1] / T_ff;
  z_b_ref[2] = f_world[2] / T_ff;
  z_b_ref[3] = f_world[3] / T_ff;

  // Yaw reference fixed at 0 -> x_c = (1, 0, 0)
  x_c[1] = 1;
  x_c[2] = 0;
  x_c[3] = 0;
  y_b_ref_raw = cross(z_b_ref, x_c);
  y_b_ref_norm = sqrt(y_b_ref_raw[1]^2 + y_b_ref_raw[2]^2 + y_b_ref_raw[3]^2 + 1e-9);
  y_b_ref[1] = y_b_ref_raw[1] / y_b_ref_norm;
  y_b_ref[2] = y_b_ref_raw[2] / y_b_ref_norm;
  y_b_ref[3] = y_b_ref_raw[3] / y_b_ref_norm;
  x_b_ref = cross(y_b_ref, z_b_ref);

  R_ref[1, 1] = x_b_ref[1];  R_ref[1, 2] = y_b_ref[1];  R_ref[1, 3] = z_b_ref[1];
  R_ref[2, 1] = x_b_ref[2];  R_ref[2, 2] = y_b_ref[2];  R_ref[2, 3] = z_b_ref[2];
  R_ref[3, 1] = x_b_ref[3];  R_ref[3, 2] = y_b_ref[3];  R_ref[3, 3] = z_b_ref[3];

  // ===== Reference body rates from jerk (NWU sign convention) =====
  // p_ff = -(m/T) · jerk · y_b ; q_ff = +(m/T) · jerk · x_b ; r_ff = 0 (yaw fixed)
  t_scaled[1] = mass * j_ref[1] / T_ff;
  t_scaled[2] = mass * j_ref[2] / T_ff;
  t_scaled[3] = mass * j_ref[3] / T_ff;
  omega_ff_b[1] = -(t_scaled[1]*y_b_ref[1] + t_scaled[2]*y_b_ref[2] + t_scaled[3]*y_b_ref[3]);
  omega_ff_b[2] =  (t_scaled[1]*x_b_ref[1] + t_scaled[2]*x_b_ref[2] + t_scaled[3]*x_b_ref[3]);
  omega_ff_b[3] = 0;

  // ===== Lie algebra error ζ = log_SE_2(3)(X^-1 X_ref) =====
  // η_R = R^T · R_ref ; η_v = R^T (v_ref - v) ; η_p = R^T (p_ref - p)
  // Small-angle approx: ζ_v ≈ η_v, ζ_p ≈ η_p (J_l^-1 ≈ I)
  S[1, 1] = R[1,1]*R_ref[1,1] + R[2,1]*R_ref[2,1] + R[3,1]*R_ref[3,1];
  S[1, 2] = R[1,1]*R_ref[1,2] + R[2,1]*R_ref[2,2] + R[3,1]*R_ref[3,2];
  S[1, 3] = R[1,1]*R_ref[1,3] + R[2,1]*R_ref[2,3] + R[3,1]*R_ref[3,3];
  S[2, 1] = R[1,2]*R_ref[1,1] + R[2,2]*R_ref[2,1] + R[3,2]*R_ref[3,1];
  S[2, 2] = R[1,2]*R_ref[1,2] + R[2,2]*R_ref[2,2] + R[3,2]*R_ref[3,2];
  S[2, 3] = R[1,2]*R_ref[1,3] + R[2,2]*R_ref[2,3] + R[3,2]*R_ref[3,3];
  S[3, 1] = R[1,3]*R_ref[1,1] + R[2,3]*R_ref[2,1] + R[3,3]*R_ref[3,1];
  S[3, 2] = R[1,3]*R_ref[1,2] + R[2,3]*R_ref[2,2] + R[3,3]*R_ref[3,2];
  S[3, 3] = R[1,3]*R_ref[1,3] + R[2,3]*R_ref[2,3] + R[3,3]*R_ref[3,3];

  trace_S = S[1,1] + S[2,2] + S[3,3];
  arg = 0.5 * (trace_S - 1);
  arg_safe = max(-0.9999, min(0.9999, arg));
  theta = acos(arg_safe);
  sin_theta = sin(theta);
  // log_factor = θ / (2·sinθ), with smooth fallback to 0.5 at θ→0
  log_factor = if abs(sin_theta) > 1e-4 then theta / (2 * sin_theta) else 0.5;

  zeta_R[1] = log_factor * (S[3, 2] - S[2, 3]);
  zeta_R[2] = log_factor * (S[1, 3] - S[3, 1]);
  zeta_R[3] = log_factor * (S[2, 1] - S[1, 2]);

  // Body-frame velocity and position errors (small-angle: ζ ≈ η)
  zeta_v[1] = R[1,1]*(v_ref[1] - velocity[1]) + R[2,1]*(v_ref[2] - velocity[2]) + R[3,1]*(v_ref[3] - velocity[3]);
  zeta_v[2] = R[1,2]*(v_ref[1] - velocity[1]) + R[2,2]*(v_ref[2] - velocity[2]) + R[3,2]*(v_ref[3] - velocity[3]);
  zeta_v[3] = R[1,3]*(v_ref[1] - velocity[1]) + R[2,3]*(v_ref[2] - velocity[2]) + R[3,3]*(v_ref[3] - velocity[3]);
  zeta_p[1] = R[1,1]*(p_ref[1] - position[1]) + R[2,1]*(p_ref[2] - position[2]) + R[3,1]*(p_ref[3] - position[3]);
  zeta_p[2] = R[1,2]*(p_ref[1] - position[1]) + R[2,2]*(p_ref[2] - position[2]) + R[3,2]*(p_ref[3] - position[3]);
  zeta_p[3] = R[1,3]*(p_ref[1] - position[1]) + R[2,3]*(p_ref[2] - position[2]) + R[3,3]*(p_ref[3] - position[3]);

  zeta[1] = zeta_p[1]; zeta[2] = zeta_p[2]; zeta[3] = zeta_p[3];
  zeta[4] = zeta_v[1]; zeta[5] = zeta_v[2]; zeta[6] = zeta_v[3];
  zeta[7] = zeta_R[1]; zeta[8] = zeta_R[2]; zeta[9] = zeta_R[3];

  // ===== LQR state feedback: u_ζ = -K · ζ =====
  for i in 1:4 loop
    u_zeta[i] = -sum(K[i, j] * zeta[j] for j in 1:9);
  end for;

  // ===== Total control commands (FF + FB) =====
  a_z_cmd = a_z_ff + u_zeta[1];
  T_cmd = mass * a_z_cmd;
  omega_cmd_b[1] = omega_ff_b[1] + u_zeta[2];
  omega_cmd_b[2] = omega_ff_b[2] + u_zeta[3];
  omega_cmd_b[3] = omega_ff_b[3] + u_zeta[4];

  // ===== Inner rate loop (eq 4.17 simplified) =====
  omega_err = omega_cmd_b - omega_b;
  for i in 1:3 loop
    omega_dot_cmd[i] = K_omega[i] * omega_err[i];
  end for;

  // M_b = J·ω̇_cmd + ω × (J·ω)
  Jw[1] = Ixx * omega_b[1];
  Jw[2] = Iyy * omega_b[2];
  Jw[3] = Izz * omega_b[3];
  M_b[1] = Ixx * omega_dot_cmd[1] + (omega_b[2]*Jw[3] - omega_b[3]*Jw[2]);
  M_b[2] = Iyy * omega_dot_cmd[2] + (omega_b[3]*Jw[1] - omega_b[1]*Jw[3]);
  M_b[3] = Izz * omega_dot_cmd[3] + (omega_b[1]*Jw[2] - omega_b[2]*Jw[1]);

  // ===== Motor mixing (QuadrotorSIL X-layout) =====
  //   1: front-right CCW   2: rear-left CCW
  //   3: front-left  CW    4: rear-right CW
  F_motor[1] = 0.25 * (T_cmd - M_b[1]/d - M_b[2]/d - M_b[3]/Cm);
  F_motor[2] = 0.25 * (T_cmd + M_b[1]/d + M_b[2]/d - M_b[3]/Cm);
  F_motor[3] = 0.25 * (T_cmd + M_b[1]/d - M_b[2]/d + M_b[3]/Cm);
  F_motor[4] = 0.25 * (T_cmd - M_b[1]/d + M_b[2]/d + M_b[3]/Cm);

  F_max = Ct * omega_cmd_max * omega_cmd_max;
  for i in 1:4 loop
    F_motor_safe[i] = max(0.0, min(F_max, F_motor[i]));
    omega_cmd[i] = sqrt(F_motor_safe[i] / Ct);
  end for;

end LLDIController;


// Composed: 3 position refs + 1 wind disturbance in, plant sensors out.
// Expects QuadrotorSIL patched with wind_dir_world parameter and wind_speed input.

model QuadrotorPositionWind
  parameter Real wind_dir_world[3] = {1, 0, 0} "Wind direction unit vector (NWU)";
  parameter Real td_omega_bw = 25.0 "TD bandwidth [rad/s] - increase for sharper derivative tracking";

  input Real px_ref(start = 0) "Reference X (NWU) [m]";
  input Real py_ref(start = 0) "Reference Y (NWU) [m]";
  input Real pz_ref(start = 0) "Reference Z (NWU) [m]";
  input Real wind_speed(start = 0) "Wind speed along wind_dir_world [m/s]";

  output Real position[3];
  output Real velocity[3];
  output Real quat[4];
  output Real omega_m[4];
  output Real accel[3];
  output Real gyro[3];
  output Real mag[3];

  // Expose reference derivatives for logging / debugging
  output Real v_ref_out[3];
  output Real a_ref_out[3];
  output Real j_ref_out[3];

protected
  QuadrotorSIL vehicle(wind_dir_world = wind_dir_world);
  TrackingDifferentiator td_x(omega_bw = td_omega_bw);
  TrackingDifferentiator td_y(omega_bw = td_omega_bw);
  TrackingDifferentiator td_z(omega_bw = td_omega_bw);
  LLDIController controller(
    mass = vehicle.vehicle_mass,
    g = vehicle.g,
    Ct = vehicle.Ct,
    Cm = vehicle.Cm,
    arm_length = vehicle.arm_length,
    omega_cmd_max = vehicle.motor_omega_cmd_max,
    Ixx = vehicle.vehicle_ixx,
    Iyy = vehicle.vehicle_iyy,
    Izz = vehicle.vehicle_izz
  );

equation
  // Tracking differentiators on each position axis
  td_x.x_in = px_ref;
  td_y.x_in = py_ref;
  td_z.x_in = pz_ref;

  // Wire TD outputs into LLDI controller reference inputs
  controller.p_ref[1] = px_ref;
  controller.p_ref[2] = py_ref;
  controller.p_ref[3] = pz_ref;
  controller.v_ref[1] = td_x.v_hat;
  controller.v_ref[2] = td_y.v_hat;
  controller.v_ref[3] = td_z.v_hat;
  controller.a_ref[1] = td_x.a_hat;
  controller.a_ref[2] = td_y.a_hat;
  controller.a_ref[3] = td_z.a_hat;
  controller.j_ref[1] = td_x.j_hat;
  controller.j_ref[2] = td_y.j_hat;
  controller.j_ref[3] = td_z.j_hat;

  // Plant <-> controller
  controller.position = vehicle.position;
  controller.velocity = vehicle.velocity;
  controller.quat = vehicle.quat;
  controller.gyro = vehicle.gyro;
  vehicle.omega_cmd = controller.omega_cmd;

  vehicle.wind_speed = wind_speed;

  // Surface outputs
  position = vehicle.position;
  velocity = vehicle.velocity;
  quat = vehicle.quat;
  omega_m = vehicle.omega_m;
  accel = vehicle.accel;
  gyro = vehicle.gyro;
  mag = vehicle.mag;

  v_ref_out[1] = td_x.v_hat; v_ref_out[2] = td_y.v_hat; v_ref_out[3] = td_z.v_hat;
  a_ref_out[1] = td_x.a_hat; a_ref_out[2] = td_y.a_hat; a_ref_out[3] = td_z.a_hat;
  j_ref_out[1] = td_x.j_hat; j_ref_out[2] = td_y.j_hat; j_ref_out[3] = td_z.j_hat;
end QuadrotorPositionWind;
