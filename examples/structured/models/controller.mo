// Controller model: PD controller with feedforward
// Computes control force based on tracking error

model Controller
  parameter Real kp = 10.0 "Proportional gain";
  parameter Real kd = 2.0 "Derivative gain";

  input Real x "Current position [m]";
  input Real v "Current velocity [m/s]";
  input Real x_ref "Reference position [m]";
  input Real v_ref "Reference velocity [m/s]";
  input Real u_ff "Feedforward control [N]";

  output Real F "Control force [N]";
  output Real e "Position error [m]";
  output Real ev "Velocity error [m/s]";

equation
  e = x - x_ref;
  ev = v - v_ref;
  F = u_ff - kp*e - kd*ev;

end Controller;
