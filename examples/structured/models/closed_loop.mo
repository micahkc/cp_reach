// Closed-loop system: Plant + Controller
// Wires together plant dynamics and PD controller
// External inputs: reference trajectory (x_ref, v_ref), feedforward (u_ff), disturbance (d)
// Outputs: tracking errors (e, ev)

model ClosedLoop
  // Plant parameters
  parameter Real m = 1.0 "Mass [kg]";
  parameter Real c = 0.2 "Damping coefficient [N*s/m]";
  parameter Real k = 1.0 "Spring constant [N/m]";

  // Controller parameters (same as MassSpringPD basic example)
  parameter Real kp = 2.0 "Proportional gain";
  parameter Real kd = 0.8 "Derivative gain";

  // External inputs (reference trajectory is an input, not internal states)
  input Real x_ref "Reference position [m]";
  input Real v_ref "Reference velocity [m/s]";
  input Real u_ff "Feedforward control [N]";
  input Real d "Disturbance force [N]";

  // Plant states
  Real x(start = 0) "Position [m]";
  Real v(start = 0) "Velocity [m/s]";

  // Controller outputs (algebraic)
  Real F "Control force [N]";
  Real e "Position error [m]";
  Real ev "Velocity error [m/s]";

equation
  // Error computation
  e = x - x_ref;
  ev = v - v_ref;

  // Controller: PD with feedforward
  F = u_ff - kp*e - kd*ev;

  // Plant dynamics
  der(x) = v;
  der(v) = (F + d - c*v - k*x) / m;

end ClosedLoop;
