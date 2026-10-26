function result = simulate_nmpc(cfg)
%SIMULATE_NMPC Measured state -> optimize -> first input -> actual plant.
ref = make_reference(cfg); K = ref.K; N = cfg.Np;
X = nan(5,K+1); U = nan(2,K); Zlog = nan(2,K); Uref = nan(2,K);
T0 = equilibrium_torque(14.5,cfg.p);
X(:,1) = [0.4;0;14.5;0;T0];
zprev = [T0;0]./cfg.inputScale;
warm = repmat(zprev,1,N);
actual = cfg.p; actual.m = cfg.p.m*cfg.actualMassFactor;
lb = -ones(2*N,1); ub = ones(2*N,1);
logData = struct('time',nan(K,1),'objective',nan(K,1),'initialCost',nan(K,1), ...
    'candidateObjective',nan(K,1),'status',nan(K,1),'exitflag',nan(K,1), ...
    'iterations',nan(K,1),'funcCount',nan(K,1),'firstorderopt',nan(K,1), ...
    'message',{cell(K,1)});
if strcmpi(cfg.solver,'fmincon')
    if exist('fmincon','file')~=2 || ~license('test','Optimization_Toolbox')
        error('fmincon requires Optimization Toolbox. Use solver=projected-bfgs.');
    end
    options = optimoptions('fmincon','Algorithm','sqp','Display','none', ...
        'SpecifyObjectiveGradient',true,'MaxIterations',cfg.maxIter, ...
        'MaxFunctionEvaluations',10000,'OptimalityTolerance',cfg.gradTol, ...
        'StepTolerance',cfg.stepTol,'ConstraintTolerance',cfg.boundTol);
elseif ~strcmpi(cfg.solver,'projected-bfgs')
    error('Unknown solver. Choose projected-bfgs or fmincon.');
end
completed = 0; termination = 'Completed requested duration';
for k = 1:K
    Yref = ref.Y(:,k:k+N);
    Zref = [equilibrium_torque(Yref(3,2:end),cfg.p)/cfg.inputScale(1); zeros(1,N)];
    objective = @(z) nmpc_objective(z,X(:,k),zprev,Yref,Zref,cfg);
    clock = tic;
    Jwarm = objective(warm(:));
    if strcmpi(cfg.solver,'fmincon')
        [candidate,Jcandidate,exitflag,info] = fmincon(objective,warm(:), ...
            [],[],[],[],lb,ub,[],options);
    else
        [candidate,Jcandidate,info] = box_bfgs(objective,warm(:),lb,ub,cfg);
        exitflag = info.exitflag;
    end
    candidate = reshape(candidate,2,N);
    [Z,status] = choose_command(candidate,Jcandidate,exitflag,warm,Jwarm,cfg);
    Jused = Jcandidate;
    if status==3, Jused = Jwarm; end
    solveTime = toc(clock); % warm-cost evaluation + optimizer + selection.
    U(:,k) = cfg.inputScale.*Z(:,1);
    Zlog(:,k) = Z(:,1); Uref(:,k) = Zref(:,1);
    logData.time(k) = solveTime; logData.objective(k) = Jused;
    logData.initialCost(k) = Jwarm; logData.candidateObjective(k) = Jcandidate;
    logData.status(k) = status; logData.exitflag(k) = exitflag;
    logData.iterations(k) = info.iterations; logData.funcCount(k) = info.funcCount;
    logData.firstorderopt(k) = info.firstorderopt; logData.message{k} = info.message;
    X(:,k+1) = vehicle_step(X(:,k),U(:,k),actual,cfg.dt);
    completed = k;
    if any(~isfinite(X(:,k+1))) || X(3,k+1)<=0
        termination = sprintf('Stopped: actual state leaves valid forward-speed domain at t=%g s',k*cfg.dt);
        warning('NMPC:PlantDomain','%s',termination); break
    end
    zprev = Z(:,1);
    warm = [Z(:,2:end),Z(:,end)];
    if cfg.progressEvery>0 && (k==1 || mod(k,cfg.progressEvery)==0 || k==K)
        fprintf('  %4d/%4d steps; t=%6.1f s; position error=%8.4f m; solve=%.3f s\n', ...
            k,K,k*cfg.dt,norm(X(1:2,k+1)-ref.Y(1:2,k+1)),solveTime);
    end
end
result.cfg = cfg; result.ref = ref;
result.t = (0:completed)*cfg.dt;
result.X = X(:,1:completed+1); result.U = U(:,1:completed);
result.Z = Zlog(:,1:completed); result.Zref = Uref(:,1:completed);
fields = fieldnames(logData);
for i = 1:numel(fields), logData.(fields{i}) = logData.(fields{i})(1:completed,:); end
result.log = logData;
result.initialNormalizedInput = [T0;0]./cfg.inputScale;
result.completed = completed==K;
result.termination = termination;
result.metrics = compute_metrics(result);
end
