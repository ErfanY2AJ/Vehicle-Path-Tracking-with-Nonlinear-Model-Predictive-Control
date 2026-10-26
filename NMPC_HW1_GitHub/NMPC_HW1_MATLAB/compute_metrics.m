function metrics = compute_metrics(result)
%COMPUTE_METRICS Radial position error; scalar speed error; normalized inputs.
K = size(result.U,2);
e = result.X(1:3,:)-result.ref.Y(:,1:K+1);
positionError = sqrt(sum(e(1:2,:).^2,1));
speedError = e(3,:);
dZ = result.Z-[result.initialNormalizedInput,result.Z(:,1:end-1)];
effort = sum((result.Z-result.Zref).^2,1);
variation = sum(dZ.^2,1);
windows = {true(1,K+1),result.t>=5-1e-10};
names = {'full','after5s'};
for j = 1:2
    mask = windows{j}; inputMask = mask(1:end-1);
    m = struct;
    m.stateSamples = nnz(mask); m.inputSamples = nnz(inputMask);
    m.positionRMSE_m = sqrt(mean(positionError(mask).^2));
    m.positionMAE_m = mean(positionError(mask));
    m.speedRMSE_mps = sqrt(mean(speedError(mask).^2));
    m.speedMAE_mps = mean(abs(speedError(mask)));
    m.Eu = mean(effort(inputMask)); m.EdeltaU = mean(variation(inputMask));
    m.meanSolveTime_s = mean(result.log.time(inputMask));
    if any(inputMask), m.maxSolveTime_s = max(result.log.time(inputMask));
    else, m.maxSolveTime_s = NaN; end
    m.converged = nnz(result.log.status(inputMask)==1);
    m.feasibleNotConverged = nnz(result.log.status(inputMask)==2);
    m.fallback = nnz(result.log.status(inputMask)==3);
    metrics.(names{j}) = m;
end
metrics.boundViolations = nnz(any(abs(result.U)>result.cfg.inputScale+result.cfg.boundTol,1));
metrics.nonfiniteAppliedInputs = nnz(any(~isfinite(result.U),1));
metrics.referencePeakLateralAcceleration_mps2 = max(abs( ...
    result.ref.Y(3,1:K+1).*result.ref.yawRate(1:K+1)));
metrics.actualPeakKinematicLateralAcceleration_mps2 = max(abs( ...
    result.X(3,1:K).^2/result.cfg.p.L.*tan(result.U(2,:))));
% Neither lateral-acceleration metric is an enforced safety constraint.
end
