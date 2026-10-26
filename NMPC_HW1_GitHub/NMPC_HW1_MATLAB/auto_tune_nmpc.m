function [bestCfg,report] = auto_tune_nmpc(mode,overrides)
%AUTO_TUNE_NMPC Automatic OFFLINE tuning; no additional toolbox is required.
% [bestCfg,report]=auto_tune_nmpc('quick');    % Small end-to-end preview
% [bestCfg,report]=auto_tune_nmpc('standard'); % 18 candidates + longer checks
% opts=struct('maxEvaluations',30); auto_tune_nmpc('standard',opts)
% Method: stratified global exploration followed by bounded local refinement.
% This is not Bayesian optimization and does not guarantee a global optimum.
if nargin<1, mode='standard'; end
if nargin<2, overrides=struct; end
here=fileparts(mfilename('fullpath')); addpath(here);
opt=nmpc_tuning_options(mode);
fields=fieldnames(overrides);
for i=1:numel(fields)
    if ~isfield(opt,fields{i}), error('Unknown tuning option: %s',fields{i}); end
    opt.(fields{i})=overrides.(fields{i});
end
assert(opt.maxEvaluations>=2 && opt.maxEvaluations==round(opt.maxEvaluations));
assert(opt.trainDuration>5 && opt.validationDuration>=opt.trainDuration);
assert(opt.scaleLimit>1 && all(opt.NpValues>=1) && all(opt.NpValues==round(opt.NpValues)));
assert(all([opt.positionFullScale,opt.speedFullScale,opt.positionPostScale, ...
    opt.speedPostScale,opt.effortScale,opt.variationScale]>0));
assert(opt.timeWeight>=0 && opt.nonconvergenceWeight>=0);
base=nmpc_config(); base.showFigures=false; base.saveFigures=false; base.progressEvery=0;
assert(any(opt.NpValues==base.Np),'Candidate horizons must include the baseline Np.');
oldRng=rng; cleanup=onCleanup(@()rng(oldRng)); %#ok<NASGU>
rng(opt.seed);
root=fullfile(base.outputRoot,['auto_tune_' datestr(now,'yyyymmdd_HHMMSS')]);
folder=root; i=1;
while exist(folder,'dir'), folder=sprintf('%s_%d',root,i); i=i+1; end
mkdir(folder);
% Warm up numerical code before comparing empirical timings.
warm=base; warm.Tsim=1; simulate_nmpc(warm);
nExplore=min(opt.maxEvaluations,max(3,ceil(0.65*opt.maxEvaluations)));
initial=zeros(nExplore,5); initial(1,:)=[base.Np,0,0,0,0];
others=opt.NpValues(opt.NpValues~=base.Np);
order=[others,base.Np];
for i=2:nExplore, initial(i,1)=order(1+mod(i-2,numel(order))); end
% Latin-hypercube-like strata, built from randperm rather than a toolbox.
for d=2:5
    initial(2:end,d)=2*((randperm(nExplore-1).'-rand(nExplore-1,1))/(nExplore-1))-1;
end
bestTheta=initial(1,:); bestScore=Inf; bestTrial=1; history=table;
trainingNames={'nominal_mild','nominal_challenging','mass_plus10pct'};
fprintf('Automatic tuning: %d candidates x 3 cases x %g seconds.\n', ...
    opt.maxEvaluations,opt.trainDuration);
fprintf('Reference: %s\nFixed dt=%g; solver=%s; gradTol=%g; stepTol=%g\n', ...
    base.referenceLabel,base.dt,base.solver,base.gradTol,base.stepTol);
for trial=1:opt.maxEvaluations
    if trial<=nExplore
        theta=initial(trial,:);
    else
        theta=bestTheta;
        radius=0.45/sqrt(trial-nExplore);
        theta(2:5)=min(1,max(-1,bestTheta(2:5)+radius*randn(1,4)));
        [~,idx]=min(abs(opt.NpValues-bestTheta(1)));
        idx=min(numel(opt.NpValues),max(1,idx+randi(3)-2));
        theta(1)=opt.NpValues(idx);
    end
    candidate=nmpc_tuning_decode(theta,base,opt);
    caseScore=inf(1,3); problems=cell(1,3); elapsed=tic;
    for j=1:3
        c=scenario(candidate,j,opt.trainDuration);
        r=simulate_nmpc(c); % Genuine implementation errors remain visible.
        caseScore(j)=nmpc_tuning_score(r,opt);
        problems{j}=r.termination;
    end
    score=mean(caseScore);
    if score<bestScore
        bestScore=score; bestTheta=theta; bestTrial=trial;
    end
    f=opt.scaleLimit.^theta(2:5);
    row=table(trial,theta(1),f(1),f(2),f(3),f(4),score,caseScore(1), ...
        caseScore(2),caseScore(3),toc(elapsed), ...
        'VariableNames',{'Trial','Np','SpeedWeightScale','RScale','TorqueRateScale', ...
        'SteeringRateScale','Score','MildScore','ChallengingScore','MassScore','Elapsed_s'});
    if trial==1, history=row; else, history=[history;row]; end %#ok<AGROW>
    fprintf('  Trial %2d/%2d: Np=%2d, score=%9.5g; best=%9.5g\n', ...
        trial,opt.maxEvaluations,theta(1),score,bestScore);
    candidateCfg=nmpc_tuning_decode(bestTheta,base,opt);
    writetable(history,fullfile(folder,'search_history.csv'));
    save(fullfile(folder,'checkpoint.mat'),'history','candidateCfg','bestTheta', ...
        'bestScore','bestTrial','opt','trainingNames','problems');
end
if ~isfinite(bestScore)
    error('All candidate simulations were invalid. Inspect the saved checkpoint.');
end
% Selection gate: compare baseline and winner over a longer interval.
% Fourth case (alpha=.6, mass+5%) was not used in the search. This is a
% validation/acceptance check, not an unbiased final statistical test set.
fprintf('\nLonger validation: %g s, 4 cases, baseline versus candidate.\n',opt.validationDuration);
comparison=table; baselineScores=zeros(4,1); candidateScores=zeros(4,1);
validationNames=[trainingNames,{'intermediate_path_mass_plus5pct'}];
for j=1:4
    cb=scenario(base,j,opt.validationDuration);
    cc=scenario(candidateCfg,j,opt.validationDuration);
    if bestTrial==1
        rb=simulate_nmpc(cb); rc=rb;
    elseif mod(j,2)==1
        rb=simulate_nmpc(cb); rc=simulate_nmpc(cc);
    else
        rc=simulate_nmpc(cc); rb=simulate_nmpc(cb);
    end
    baselineScores(j)=nmpc_tuning_score(rb,opt);
    candidateScores(j)=nmpc_tuning_score(rc,opt);
    b=rb.metrics.after5s; c=rc.metrics.after5s;
    row=table(string(validationNames{j}),baselineScores(j),candidateScores(j), ...
        b.positionRMSE_m,c.positionRMSE_m,b.speedRMSE_mps,c.speedRMSE_mps, ...
        rb.metrics.full.meanSolveTime_s,rc.metrics.full.meanSolveTime_s, ...
        'VariableNames',{'Case','BaselineScore','CandidateScore','BaselinePositionRMSE_m', ...
        'CandidatePositionRMSE_m','BaselineSpeedRMSE_mps','CandidateSpeedRMSE_mps', ...
        'BaselineMeanSolve_s','CandidateMeanSolve_s'});
    if j==1, comparison=row; else, comparison=[comparison;row]; end %#ok<AGROW>
    save(fullfile(folder,sprintf('validation_%d.mat',j)),'rb','rc');
end
accepted=bestTrial~=1 && isfinite(mean(candidateScores)) && mean(candidateScores)<mean(baselineScores);
bestCfg=base;
if accepted, bestCfg=candidateCfg; end
bestCfg.Tsim=100; bestCfg.progressEvery=50; bestCfg.showFigures=true; bestCfg.saveFigures=true;
report=struct('history',history,'comparison',comparison,'candidateCfg',candidateCfg, ...
    'accepted',accepted,'bestTrainingTrial',bestTrial,'bestTrainingScore',bestScore, ...
    'baselineValidationScore',mean(baselineScores), ...
    'candidateValidationScore',mean(candidateScores),'options',opt,'folder',folder);
writetable(comparison,fullfile(folder,'validation_comparison.csv'));
save(fullfile(folder,'tuning_result.mat'),'bestCfg','report');
disp(comparison);
if accepted
    fprintf('Candidate accepted by the fixed validation objective. Np=%d\n',bestCfg.Np);
else
    fprintf('Candidate did not improve validation score. Returning baseline settings.\n');
end
fprintf('Saved search and validation results: %s\n',folder);
end
function c=scenario(base,index,duration)
c=base; c.Tsim=duration; c.showFigures=false; c.saveFigures=false; c.progressEvery=0;
c.actualMassFactor=1; c.alpha=0.3;
switch index
    case 1
    case 2, c.alpha=1;
    case 3, c.actualMassFactor=1.1;
    case 4, c.alpha=0.6; c.actualMassFactor=1.05;
end
end
