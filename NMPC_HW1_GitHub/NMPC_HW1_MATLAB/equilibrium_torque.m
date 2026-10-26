function T = equilibrium_torque(v,p)
%EQUILIBRIUM_TORQUE Torque balancing the resistance at forward speed v.
T = (p.R/p.eta)*(p.CA*v.^2 + p.m*p.g*p.fr);
end
