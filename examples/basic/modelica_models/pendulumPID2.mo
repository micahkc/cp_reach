within ;

// ============================================================
// MAIN MODEL (what Rumoca should see as "the" system)
// ============================================================
model pendulumPID2
  // Shared physical / control parameters
  parameter Real m = 1.0 "Mass";
  parameter Real l = 1.0 "Length of pendulum";
  parameter Real g = 9.81 "Gravitational acceleration";
  parameter Real c = 0.1 "Damping coefficient";

  parameter Real Kp = 5.0 "Proportional gain";
  parameter Real Ki = 1.0 "Integral gain";
  parameter Real Kd = 0.5 "Derivative gain";

  // ===== External Inputs (for reachability / simulation) =====
  input Real theta_ref "Reference angle";
  input Real dtheta_ref "Reference angular rate";
  input Real ddtheta_ref "Reference angular acceleration";
  input Real d = 0 "Additive disturbance torque";

  // ===== Subsystems =====
  PlantDynamics plant(
    m = m,
    l = l,
    g = g,
    c = c) "True pendulum";

  PIDController ctrl(
    Kp = Kp,
    Ki = Ki,
    Kd = Kd,
    m = m,
    l = l,
    g = g,
    c = c) "PID controller with feedforward";

  // ===== Outputs: Error signals (for cp_reach) =====
  output Real e_theta "Angle tracking error (theta_ref - theta)";
  output Real e_omega "Angular rate tracking error (dtheta_ref - omega)";
equation
  // Plant input
  plant.u = ctrl.u;
  plant.d = d;

  // Controller inputs
  ctrl.theta_ref   = theta_ref;
  ctrl.dtheta_ref  = dtheta_ref;
  ctrl.ddtheta_ref = ddtheta_ref;

  ctrl.theta = plant.theta;
  ctrl.omega = plant.omega;

  // Error outputs
  e_theta = theta_ref  - plant.theta;
  e_omega = dtheta_ref - plant.omega;
end pendulumPID2;


// ============================================================
// PLANT DYNAMICS (TRUE PENDULUM)
// ============================================================
model PlantDynamics
  parameter Real m = 1.0 "Mass";
  parameter Real l = 1.0 "Length of pendulum";
  parameter Real g = 9.81 "Gravitational acceleration";
  parameter Real c = 0.1 "Damping coefficient";

  input Real u "Control torque";
  input Real d = 0 "Additive disturbance torque";

  Real theta(start = 0) "Angle";
  Real omega(start = 0) "Angular rate";
equation
  der(theta) = omega;
  der(omega) = -(g/l)*sin(theta)
               - c/(m*l^2)*omega
               + (u + d)/(m*l^2);
end PlantDynamics;


// ============================================================
// PID CONTROLLER (WITH FEEDFORWARD)
// ============================================================
model PIDController
  // PID gains
  parameter Real Kp = 5.0 "Proportional gain";
  parameter Real Ki = 1.0 "Integral gain";
  parameter Real Kd = 0.5 "Derivative gain";

  // Physical params for feedforward term
  parameter Real m = 1.0 "Mass";
  parameter Real l = 1.0 "Length of pendulum";
  parameter Real g = 9.81 "Gravitational acceleration";
  parameter Real c = 0.1 "Damping coefficient";

  // Reference inputs
  input Real theta_ref "Reference angle";
  input Real dtheta_ref "Reference angular rate";
  input Real ddtheta_ref "Reference angular acceleration";

  // Measured plant states
  input Real theta "Measured angle";
  input Real omega "Measured angular rate";

  // Control output
  output Real u "Total control torque";

protected
  Real xi "Integral of angle error";
  Real e "Angle error";
  Real e_omega "Angular rate error";

  Real u_ff "Feedforward torque";
  Real u_pid "PID correction torque";
equation
  // Errors
  e       = theta_ref - theta;
  e_omega = dtheta_ref - omega;

  // Integral of angle error
  der(xi) = e;

  // Feedforward from reference dynamics:
  //   tau_ref = I*ddtheta_ref + c*dtheta_ref + m*g*l*sin(theta_ref)
  u_ff = m*l^2*ddtheta_ref + c*dtheta_ref + m*g*l*sin(theta_ref);

  // PID correction (derivative on measurement: uses omega)
  // You could instead use e_omega if you prefer derivative of error.
  u_pid = Kp*e + Ki*xi - Kd*omega;

  // Total control
  u = u_ff + u_pid;
end PIDController;
