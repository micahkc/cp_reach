// Closed-loop system using component composition
// This structure allows rumoca to preserve component prefixes in flattening

model MassSpringDamper "Plant: Mass-spring-damper system"
  parameter Real m = 1.0 "Mass [kg]";
  parameter Real c = 0.2 "Damping coefficient [N*s/m]";
  parameter Real k = 1.0 "Spring constant [N/m]";

  input Real F "Applied force [N]";
  input Real d "Disturbance force [N]";

  Real x(start = 0) "Position [m]";
  Real v(start = 0) "Velocity [m/s]";

  output Real y_x "Position output";
  output Real y_v "Velocity output";
equation
  der(x) = v;
  der(v) = (F + d - c*v - k*x) / m;
  y_x = x;
  y_v = v;
end MassSpringDamper;

model PDController "Controller: PD with feedforward"
  parameter Real kp = 10.0 "Proportional gain";
  parameter Real kd = 2.0 "Derivative gain";

  input Real x_ref "Reference position";
  input Real v_ref "Reference velocity";
  input Real x_meas "Measured position";
  input Real v_meas "Measured velocity";
  input Real u_ff "Feedforward";

  output Real F "Control force";

  Real e "Position error";
  Real ev "Velocity error";
equation
  e = x_meas - x_ref;
  ev = v_meas - v_ref;
  F = u_ff - kp*e - kd*ev;
end PDController;

model ClosedLoopComposed "Composed closed-loop system"
  // Component instances - these create the prefixes!
  MassSpringDamper plant;
  PDController controller;

  // External inputs
  input Real x_ref "Reference position";
  input Real v_ref "Reference velocity";
  input Real u_ff "Feedforward control";
  input Real d "Disturbance";

  // Outputs (tracking errors)
  output Real e "Position error";
  output Real ev "Velocity error";
equation
  // Wire plant outputs to controller inputs
  controller.x_meas = plant.y_x;
  controller.v_meas = plant.y_v;

  // Wire controller output to plant input
  plant.F = controller.F;

  // External connections
  controller.x_ref = x_ref;
  controller.v_ref = v_ref;
  controller.u_ff = u_ff;
  plant.d = d;

  // Output the errors
  e = controller.e;
  ev = controller.ev;
end ClosedLoopComposed;
