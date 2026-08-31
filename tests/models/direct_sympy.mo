model DirectSympy
  parameter Real k = 0.5;
  input Real d = 0.0;
  input Real r = 0.0;
  Real x(start = 1.0);
  output Real e;
equation
  der(x) = -k*x + d;
  e = x - r;
end DirectSympy;
