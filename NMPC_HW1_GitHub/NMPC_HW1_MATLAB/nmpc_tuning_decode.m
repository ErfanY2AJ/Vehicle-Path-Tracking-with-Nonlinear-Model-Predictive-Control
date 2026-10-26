function cfg = nmpc_tuning_decode(theta,base,opt)
%NMPC_TUNING_DECODE [Np, log-coordinate x4] -> bounded controller parameters.
% theta(2:5) in [-1,1] maps to multipliers [1/scaleLimit, scaleLimit].
% Coordinates: speed Q/P; both R entries; torque S; steering S.
% Position Q/P stay fixed, removing the irrelevant common cost-scale freedom.
theta=theta(:).';
assert(numel(theta)==5 && all(isfinite(theta)));
assert(all(abs(theta(2:5))<=1+1e-12));
assert(any(theta(1)==opt.NpValues),'Np is outside the candidate set');
factors=opt.scaleLimit.^theta(2:5);
cfg=base;
cfg.Np=theta(1);
cfg.Q(3,3)=base.Q(3,3)*factors(1);
cfg.P(3,3)=base.P(3,3)*factors(1);
cfg.R=base.R*factors(2);
cfg.S(1,1)=base.S(1,1)*factors(3);
cfg.S(2,2)=base.S(2,2)*factors(4);
% dt, physical parameters, input scaling/bounds and solver tolerances stay fixed.
end
