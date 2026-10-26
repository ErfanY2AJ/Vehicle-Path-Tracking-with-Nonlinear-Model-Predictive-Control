function [xn,A,B] = vehicle_step(x,u,p,dt)
%VEHICLE_STEP Exact discrete mapping from Eq. (2); no ODE integrator.
% All right-hand sides use OLD states. Do not update x in place.
% A = d(x_next)/dx; B = d(x_next)/d(physical input).
x = x(:); u = u(:);
v = x(3); phi = x(4); T = x(5);
delta = u(2);
F = p.CA*v^2 + p.m*p.g*p.fr;
xn = [x(1) + dt*v*cos(phi); ...
      x(2) + dt*v*sin(phi); ...
      v + p.eta*dt/(p.m*p.R)*(T - p.R/p.eta*F); ...
      phi + dt*v/p.L*tan(delta); ...
      T + dt/p.tau*(u(1)-T)];
% Heading remains unwrapped. Speed is not clipped or artificially repaired.
if nargout > 1
    A = eye(5);
    A(1,3) = dt*cos(phi); A(1,4) = -dt*v*sin(phi);
    A(2,3) = dt*sin(phi); A(2,4) = dt*v*cos(phi);
    A(3,3) = 1 - 2*dt*p.CA*v/p.m;
    A(3,5) = p.eta*dt/(p.m*p.R);
    A(4,3) = dt/p.L*tan(delta);
    A(5,5) = 1 - dt/p.tau;
    B = zeros(5,2);
    B(4,2) = dt*v/(p.L*cos(delta)^2);
    B(5,1) = dt/p.tau;
end
end
