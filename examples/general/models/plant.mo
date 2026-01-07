// Plant model: Mass-spring-damper system
// States: position (x), velocity (v)
// Inputs: force (F), disturbance (d)

model Plant
  parameter Real m = 1.0 "Mass [kg]";
  parameter Real c = 0.2 "Damping coefficient [N*s/m]";
  parameter Real k = 1.0 "Spring constant [N/m]";

  input Real F "Control force [N]";
  input Real d "Disturbance force [N]";

  output Real x(start = 0) "Position [m]";
  output Real v(start = 0) "Velocity [m/s]";

equation
  der(x) = v;
  der(v) = (F + d - c*v - k*x) / m;

end Plant;
