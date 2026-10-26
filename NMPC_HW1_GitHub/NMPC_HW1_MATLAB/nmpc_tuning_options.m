function opt = nmpc_tuning_options(mode)
%NMPC_TUNING_OPTIONS Fixed OUTER objective; these are preferences, not MPC weights.
% No toolbox needed. Training and validation durations must exceed 5 s.
if nargin<1, mode='standard'; end
opt.seed=42;
opt.maxEvaluations=18;
opt.trainDuration=30;
opt.validationDuration=100;
opt.NpValues=[10 15 20 25 30];
opt.scaleLimit=4; % Each searched weight multiplier lies in [1/4,4].
% Fixed normalization targets in physical units, used for EVERY candidate.
opt.positionFullScale=0.10;   % m
opt.speedFullScale=0.10;      % m/s
opt.positionPostScale=0.005;  % m
opt.speedPostScale=0.005;     % m/s
opt.effortScale=0.01;        % normalized-input squared magnitude
opt.variationScale=0.005;    % normalized-input squared increment
opt.timeWeight=0.2;         % Time penalty is empirical and hardware-dependent.
opt.nonconvergenceWeight=0.02;
switch lower(char(mode))
    case 'quick'
        opt.maxEvaluations=3;
        opt.trainDuration=12;
        opt.validationDuration=15;
        opt.NpValues=[10 20 30];
    case 'standard'
    otherwise, error('Tuning mode must be quick or standard.');
end
end
