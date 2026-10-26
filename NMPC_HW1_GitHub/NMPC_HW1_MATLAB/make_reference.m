function ref = make_reference(cfg)
%MAKE_REFERENCE Forward Euler with integer-indexed times, including padding.
K = round(cfg.Tsim/cfg.dt);
if abs(K*cfg.dt-cfg.Tsim) > 1e-10
    error('Tsim must be an integer multiple of dt.');
end
t = (0:K+cfg.Np)*cfg.dt; % K+Np+1 states, including the final MPC horizon.
if ~isempty(cfg.referenceCSV)
    data = readtable(cfg.referenceCSV);
    need = {'t','px','py','v','phi'};
    if ~all(ismember(need,data.Properties.VariableNames))
        error('Reference CSV requires columns: t,px,py,v,phi.');
    end
    n = numel(t);
    if height(data) < n || any(abs(data.t(1:n).'-t)>1e-9)
        error('Reference CSV must have this exact dt and K+Np+1 samples.');
    end
    ref.t = t;
    ref.X = [data.px(1:n).';data.py(1:n).';data.v(1:n).';data.phi(1:n).'];
    if any(~isfinite(ref.X(:))) || any(ref.X(3,:)<=0)
        error('Reference data must be finite and have positive speed.');
    end
    ref.yawRate = [diff(ref.X(4,:))/cfg.dt,0];
    ref.label = ['Imported reference: ' cfg.referenceCSV];
else
    X = zeros(4,numel(t)); % [px; py; v; phi]
    X(3,1) = 14;
    yawRate = cfg.alpha*cfg.yawRateFcn(t);
    if numel(yawRate) ~= numel(t) || any(~isfinite(yawRate(:)))
        error('yawRateFcn must return one finite rad/s value for each time.');
    end
    yawRate = reshape(yawRate,1,[]);
    a = double(t >= 1 & t < 2); % No repeated t=t+dt accumulation.
    for k = 1:numel(t)-1
        v = X(3,k); phi = X(4,k);
        X(:,k+1) = X(:,k) + cfg.dt*[v*cos(phi);v*sin(phi);a(k);yawRate(k)];
    end
    ref.t = t; ref.X = X; ref.yawRate = yawRate;
    ref.label = cfg.referenceLabel;
end
ref.Y = ref.X(1:3,:);
ref.K = K;
end
