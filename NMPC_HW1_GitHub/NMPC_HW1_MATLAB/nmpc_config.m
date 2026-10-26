function cfg = nmpc_config()
%NMPC_CONFIG All user-editable settings, with physical units.
% State order: [px (m); py (m); v (m/s); phi (rad); T (N*m)].
% Input order: [commanded torque (N*m); steering angle (rad)].
cfg.dt = 0.1;
cfg.Tsim = 100;
cfg.Np = 20;
cfg.alpha = 0.30;
cfg.Q = diag([2 2 3]);
cfg.P = diag([8 8 6]);
cfg.R = diag([0.2 0.3]);
cfg.S = diag([0.3 3]);
cfg.inputScale = [1965.1; 0.5];
cfg.p = struct('m',1270,'R',0.3036,'CA',0.9871, ...
    'tau',0.5107,'eta',0.96,'fr',0.01,'g',9.8,'L',2);
cfg.actualMassFactor = 1;
cfg.solver = 'projected-bfgs'; % Written from scratch; no toolbox needed.
% Optional alternative: cfg.solver = 'fmincon'; requires Optimization Toolbox.
cfg.maxIter = 200;
cfg.gradTol = 1e-6;
cfg.functionTol = 1e-9; % Diagnostic cost-stagnation threshold; does not stop BFGS.
cfg.stepTol = 1e-10;
cfg.maxLineSearch = 40;
cfg.boundTol = 1e-9; % Applied-input tolerance in PHYSICAL units, per component.
cfg.progressEvery = 50;
cfg.showFigures = true;
cfg.saveFigures = true;
cfg.outputRoot = fullfile(fileparts(mfilename('fullpath')), 'results');
% The assignment's reference_starter.py was not supplied. This explicitly
% named practice function is NOT a reconstruction of its original path.
cfg.yawRateFcn = @practice_yaw_rate;
cfg.referenceLabel = 'Practice reference (not the supplied assignment path)';
cfg.referenceCSV = ''; % Optional exact grid: t,px,py,v,phi (incl. full horizon).
end
