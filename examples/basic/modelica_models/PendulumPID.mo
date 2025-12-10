model pendulum_pid_io "Damped pendulum with PID ref and disturbance as inputs"
  // --- Plant parameters ---
  parameter Real m = 1.0 "Mass";
  parameter Real l = 1.0 "Length of pendulum";
  parameter Real g = 9.81 "Gravitational acceleration";
  parameter Real c = 0.1 "Damping coefficient";

  // --- PID parameters ---
  parameter Real Kp = 5.0 "Proportional gain";
  parameter Real Ki = 1.0 "Integral gain";
  parameter Real Kd = 0.5 "Derivative gain";

  // --- I/O ---
  input Real theta_ref "Reference angle";
  input Real dtheta_ref "Reference angular rate";
  input Real ddtheta_ref "Reference angular acceleration";
  input Real d = 0 "Additive disturbance torque";

  // --- States ---
  Real theta "Angle";
  Real omega "Angular rate";
  Real xi "Integral of angle error";

  // --- Internal ---
  Real e "Angle error";
  Real e_omega "Angular rate error";

equation
  // Plant
  der(theta) = omega;
  der(omega) = -(g/l)*sin(theta) - c/(m*l^2)*omega + (u + d)/(m*l^2);

  // PID (derivative on measurement, filtered)
  e = theta_ref - theta;
  der(xi) = e;

  // (ADDED) feedforward from reference dynamics
  u = m*l^2*ddtheta_ref + c*dtheta_ref + m*g*l*sin(theta_ref);

  // Sum feedforward + existing PID correction
  u = u + (Kp*e + Ki*xi - Kd*omega);
end pendulum_pid_io;
