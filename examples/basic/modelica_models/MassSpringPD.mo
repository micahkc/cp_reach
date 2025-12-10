model MassSpringPD
  // Mass-spring-damper with feedforward reference and PD feedback plus disturbance on velocity dynamics.
  parameter Real m = 1 "Mass";
  parameter Real c = 0.2 "Damping";
  parameter Real k = 1 "Stiffness";

  parameter Real kp = 2 "Proportional gain";
  parameter Real kd = 0.8 "Derivative gain";

  // Reference trajectory feedforward force (can be time-varying if bound to an input)
  input Real u_ff = 0 "Feedforward input for reference and actual systems";
  // Disturbance entering v-dot of the actual system
  input Real d = 0 "Disturbance on velocity derivative of actual system";

  // Reference states
  Real x_ref(start = 0) "Reference position";
  Real v_ref(start = 0) "Reference velocity";

  // Actual states
  Real x(start = 0) "Actual position";
  Real v(start = 0) "Actual velocity";

  // Feedback control
  Real e "Position error";
  Real ev "Velocity error";
  Real u_fb "PD feedback control";
equation
  // Reference dynamics: follows feedforward input only
  der(x_ref) = v_ref;
  m * der(v_ref) = u_ff - c * v_ref - k * x_ref;

  // Feedback on error
  e = x - x_ref;
  ev = v - v_ref;
  u_fb = -kp * e - kd * ev;

  // Actual dynamics: feedforward + feedback + disturbance on v-dot
  der(x) = v;
  m * der(v) = u_ff + u_fb + d - c * v - k * x;
end MassSpringPD;
