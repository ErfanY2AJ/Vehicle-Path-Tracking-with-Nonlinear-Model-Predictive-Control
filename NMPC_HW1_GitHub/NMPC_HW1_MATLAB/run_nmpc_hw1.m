function results = run_nmpc_hw1(mode,overrides)
%RUN_NMPC_HW1 Start here: results = run_nmpc_hw1('quick');
% Modes: quick, baseline, challenging, uncertainty, full.
% Example: run_nmpc_hw1('baseline',struct('solver','fmincon'))
% Example: run_nmpc_hw1('quick',struct('showFigures',false))
if nargin<1, mode = 'quick'; end
if nargin<2, overrides = struct; end
here = fileparts(mfilename('fullpath')); addpath(here);
cfg = nmpc_config();
switch lower(char(mode))
    case 'quick', cfg.Tsim = 15;
    case {'baseline','challenging','uncertainty'}
    case 'full', cfg.showFigures = false;
    otherwise, error('Mode must be quick, baseline, challenging, uncertainty or full.');
end
fields = fieldnames(overrides);
for j = 1:numel(fields)
    if ~isfield(cfg,fields{j}), error('Unknown setting: %s',fields{j}); end
    cfg.(fields{j}) = overrides.(fields{j});
end
assert(cfg.dt>0 && cfg.Np>=1 && cfg.Np==round(cfg.Np) && cfg.Tsim>0);
configs = {cfg}; names = {'nominal_mild'};
switch lower(char(mode))
    case 'challenging'
        configs{1}.alpha = 1; names{1} = 'nominal_challenging';
    case 'uncertainty'
        configs{1}.actualMassFactor = 1.1; names{1} = 'mass_plus10pct';
    case 'full'
        if ~isempty(cfg.referenceCSV)
            error(['Full sweep changes dt. Use a yawRateFcn for this sweep, ' ...
                'or run separate baseline cases with a CSV at each exact dt.']);
        end
        configs{2}=cfg; configs{2}.alpha=1; names{2}='nominal_challenging';
        configs{3}=cfg; configs{3}.actualMassFactor=1.1; names{3}='mass_plus10pct';
        configs{4}=cfg; configs{4}.Np=10; names{4}='horizon10';
        configs{5}=cfg; configs{5}.Np=30; names{5}='horizon30';
        configs{6}=cfg; configs{6}.dt=0.05; names{6}='dt005';
        configs{7}=cfg; configs{7}.dt=0.2; names{7}='dt020';
        configs{8}=cfg; configs{8}.S(2,2)=0.1*cfg.S(2,2); names{8}='steering_rate_weight_x01';
        configs{9}=cfg; configs{9}.S(2,2)=10*cfg.S(2,2); names{9}='steering_rate_weight_x10';
end
if ~exist(cfg.outputRoot,'dir'), mkdir(cfg.outputRoot); end
prefix = fullfile(cfg.outputRoot,[char(mode) '_' datestr(now,'yyyymmdd_HHMMSS')]);
folder=prefix; suffix=1;
while exist(folder,'dir'), folder=sprintf('%s_%d',prefix,suffix); suffix=suffix+1; end
mkdir(folder);
fprintf('Reference: %s\nSolver: %s\nOutput: %s\n',cfg.referenceLabel,cfg.solver,folder);
results = cell(size(configs));
for j = 1:numel(configs)
    fprintf('\nCase %d/%d: %s\n',j,numel(configs),names{j});
    r = simulate_nmpc(configs{j});
    caseFolder = fullfile(folder,names{j}); mkdir(caseFolder);
    save(fullfile(caseFolder,'result.mat'),'r','-v7');
    export_results(r,caseFolder);
    plot_results(r,caseFolder);
    m = r.metrics.full; a = r.metrics.after5s;
    row = struct('caseName',string(names{j}),'completed',r.completed, ...
        'dt_s',r.cfg.dt,'Np',r.cfg.Np,'horizon_s',r.cfg.dt*r.cfg.Np, ...
        'massFactor',r.cfg.actualMassFactor,'alpha',r.cfg.alpha, ...
        'positionRMSE_m',m.positionRMSE_m,'speedRMSE_mps',m.speedRMSE_mps, ...
        'positionMAE_m',m.positionMAE_m,'speedMAE_mps',m.speedMAE_mps, ...
        'positionRMSE_after5_m',a.positionRMSE_m,'speedRMSE_after5_mps',a.speedRMSE_mps, ...
        'positionMAE_after5_m',a.positionMAE_m,'speedMAE_after5_mps',a.speedMAE_mps, ...
        'Eu',m.Eu,'EdeltaU',m.EdeltaU,'Eu_after5',a.Eu,'EdeltaU_after5',a.EdeltaU, ...
        'meanSolve_s',m.meanSolveTime_s,'maxSolve_s',m.maxSolveTime_s, ...
        'converged',m.converged,'feasibleNotConverged',m.feasibleNotConverged, ...
        'fallback',m.fallback,'boundViolations',r.metrics.boundViolations, ...
        'peakReferenceAy_mps2',r.metrics.referencePeakLateralAcceleration_mps2);
    if j==1, summary=struct2table(row); else, summary=[summary;struct2table(row)]; end %#ok<AGROW>
    writetable(summary,fullfile(folder,'summary.csv'));
    fprintf('  RMSE position: %.6g m; speed: %.6g m/s\n',m.positionRMSE_m,m.speedRMSE_mps);
    fprintf('  Converged=%d; feasible-not-converged=%d; fallback=%d\n', ...
        m.converged,m.feasibleNotConverged,m.fallback);
    results{j}=r;
end
disp(summary);
fprintf('Saved results in: %s\n',folder);
end
