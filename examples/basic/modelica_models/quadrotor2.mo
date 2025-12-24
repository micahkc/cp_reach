model Quadrotor
    extends RigidBody6DOF;
    parameter Real l = 1.0;
    parameter Real g = 9.81;
    parameter Real m = 1.0;
    parameter Real J_x = 1;
    parameter Real J_y = 1;
    parameter Real J_z = 1;
    parameter Real J_xz = 0.0;
    parameter Real Lambda = 1; // Jx*Jz - Jxz*Jxz;
    Real x, y, h;
    Real P, Q, R;
    Real U, V, W;
    Real F_x, F_y, F_z;
    Real phi, theta, psi;
    input Real P, Q, R;


equation
    // body forces
    F_x = -(m*g)*sin(theta);
    F_y = (m*g)*sin(phi)*cos(theta);
    F_z = (m*g)*cos(phi)*cos(theta);
    // navigation equations
    der(x) = U*cos(theta)*cos(psi) + V*(-cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi)) + W*(sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(psi));
    der(y) = U*cos(theta)*sin(psi) + V*(cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi)) + W*(-sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi));
    der(h) = U*sin(theta) - V*sin(phi)*cos(theta) - W*cos(phi)*cos(theta);

    // force equations
    der(U) = R*V - Q*W + F_x/m;
    der(V) = -R*U + P*W + F_y/m;
    der(W) = Q*U - P*V + F_z/m;

    // kinematic equations
    der(phi) = P + tan(theta)*(Q*sin(phi) + R*cos(phi));
    der(theta) = Q*cos(phi) - R*sin(phi);
    der(psi) = (Q*sin(phi) + R*cos(phi))/cos(theta);


end Quadrotor;