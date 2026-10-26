function [Z,status] = choose_command(candidate,Jcandidate,exitflag,warm,Jwarm,cfg)
%CHOOSE_COMMAND Accept or fall back, without silently clipping a bad answer.
% status 1: converged; 2: feasible, not converged; 3: warm-start fallback.
valid = isequal(size(candidate),size(warm)) && ...
    all(isfinite(candidate(:))) && isfinite(Jcandidate) && ...
    all(all(abs(cfg.inputScale.*candidate) <= cfg.inputScale+cfg.boundTol));
if ~valid || (exitflag<=0 && Jcandidate>Jwarm)
    Z = warm; status = 3;
else
    Z = candidate;
    status = 2;
    if exitflag>0, status = 1; end
end
end
