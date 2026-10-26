function r = practice_yaw_rate(t)
%PRACTICE_YAW_RATE Educational yaw-rate profile in rad/s, NOT curvature.
% This function is an explicitly chosen example, not the missing official
% yaw_rate_base. Sign/rate changes exercise receding-horizon steering.
r = zeros(size(t));
mask = t >= 10 & t < 70;
r(mask) = 0.24*cos(2*pi*(t(mask)-10)/30);
mask = t >= 70 & t < 92;
r(mask) = -0.24*cos(2*pi*(t(mask)-70)/22);
end
