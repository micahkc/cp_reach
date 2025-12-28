// Closed-loop pendulum system: Plant + PID Controller
// External inputs: reference trajectory (theta_ref, omega_ref, alpha_ref), disturbance (d)
// Nonlinear dynamics due to sin(theta) term

model PendulumClosedLoop
  // Plant parameters
  parameter Real m = 1.0 "Mass [kg]";
  parameter Real l = 1.0 "Pendulum length [m]";
  parameter Real g = 9.81 "Gravitational acceleration [m/s^2]";
  parameter Real c = 0.1 "Damping coefficient [N*m*s/rad]";

  // PID Controller parameters (matching pendulum_example.ipynb)
  parameter Real Kp = 5.0 "Proportional gain";
  parameter Real Ki = 1.0 "Integral gain";
  parameter Real Kd = 0.5 "Derivative gain";

  // External inputs (reference trajectory)
  input Real theta_ref "Reference angle [rad]";
  input Real omega_ref "Reference angular velocity [rad/s]";
  input Real alpha_ref "Reference angular acceleration [rad/s^2]";
  input Real u_ff "Feedforward torque [N*m]";
  input Real d "Disturbance torque [N*m]";

  // Plant states
  Real theta(start = 0) "Angle [rad]";
  Real omega(start = 0) "Angular velocity [rad/s]";
  Real xi(start = 0) "Integral of angle error [rad*s]";

  // Controller outputs (algebraic)
  Real e "Angle error [rad]";
  Real u "Total control torque [N*m]";

equation
  // Error computation
  e = theta_ref - theta;

  // PID Controller with feedforward
  // u_ff should be computed externally as: m*l^2*alpha_ref + c*omega_ref + m*g*l*sin(theta_ref)
  u = u_ff + Kp*e + Ki*xi - Kd*omega;

  // Plant dynamics (nonlinear pendulum)
  der(theta) = omega;
  der(omega) = -(g/l)*sin(theta) - (c/(m*l^2))*omega + (u + d)/(m*l^2);
  der(xi) = e;

end PendulumClosedLoop;
