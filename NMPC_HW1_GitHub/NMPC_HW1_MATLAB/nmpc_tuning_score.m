function [score,parts] = nmpc_tuning_score(r,opt)
%NMPC_TUNING_SCORE Compare observed performance, NEVER the reweighted MPC J.
% Completing an invalid/shortened rollout or violating bounds is unacceptable.
score=Inf; parts=nan(1,4);
if ~r.completed || r.metrics.boundViolations>0 || ...
        r.metrics.nonfiniteAppliedInputs>0 || any(~isfinite(r.X(:))) || any(r.X(3,:)<=0)
    return
end
a=r.metrics.full; b=r.metrics.after5s;
values=[a.positionRMSE_m,a.speedRMSE_mps,b.positionRMSE_m,b.speedRMSE_mps, ...
    a.Eu,a.EdeltaU,r.log.time(:).'];
if any(~isfinite(values)), return; end
tracking=(a.positionRMSE_m/opt.positionFullScale)^2 + ...
    (a.speedRMSE_mps/opt.speedFullScale)^2 + ...
    (b.positionRMSE_m/opt.positionPostScale)^2 + ...
    (b.speedRMSE_mps/opt.speedPostScale)^2;
commands=0.02*(a.Eu/opt.effortScale+a.EdeltaU/opt.variationScale);
times=sort(r.log.time(:));
p95=times(max(1,ceil(0.95*numel(times))));
% Soft time costs include average, slowest 5% and measured deadline misses.
timing=opt.timeWeight*((mean(times)/r.cfg.dt)^2 + ...
    max(0,p95/r.cfg.dt-1)^2 + max(0,max(times)/r.cfg.dt-1)^2 + ...
    mean(times>r.cfg.dt));
statuses=r.log.status(:);
solver=opt.nonconvergenceWeight*mean(statuses==2)+10*mean(statuses==3);
parts=[tracking,commands,timing,solver];
score=sum(parts);
end
