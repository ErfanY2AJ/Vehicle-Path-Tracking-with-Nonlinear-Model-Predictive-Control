function [z,J,info] = box_bfgs(fun,z0,lb,ub,opt)
%BOX_BFGS Teaching implementation of projected, bound-constrained BFGS.
% This is NOT the L-BFGS-B algorithm. It stores a dense inverse Hessian,
% freezes outward-pointing active components, and uses projected Armijo steps.
% Numerical stagnation is explicitly reported as non-convergence.
z = min(max(z0(:),lb),ub);
[J,g] = fun(z);
n = numel(z); H = eye(n); evaluations = 1;
info.exitflag = 0; info.message = 'Maximum iterations reached';
info.iterations = 0; info.costStagnated = false;
for it = 1:opt.maxIter
    pg = z-min(max(z-g,lb),ub);
    if norm(pg,inf) <= opt.gradTol
        info.exitflag = 1; info.message = 'Projected gradient tolerance satisfied';
        break
    end
    if ~isfinite(J) || any(~isfinite(g))
        info.exitflag = -2; info.message = 'Non-finite objective or gradient'; break
    end
    active = (z<=lb+1e-12 & g>0) | (z>=ub-1e-12 & g<0);
    free = ~active;
    direction = zeros(n,1);
    direction(free) = -H(free,free)*g(free);
    if ~all(isfinite(direction)) || g.'*direction >= -1e-14*norm(g)*norm(direction)
        direction = -g; direction(active) = 0; H = eye(n);
    end
    accepted = false; a = 1;
    for ls = 1:opt.maxLineSearch
        trial = min(max(z+a*direction,lb),ub);
        step = trial-z;
        [Jtrial,gtrial] = fun(trial); evaluations = evaluations+1;
        if all(isfinite(gtrial)) && isfinite(Jtrial) && ...
                g.'*step < 0 && Jtrial <= J+1e-4*(g.'*step)
            accepted = true; break
        end
        a = 0.5*a;
    end
    if ~accepted
        info.exitflag = -1; info.message = 'Projected line search stalled'; break
    end
    y = gtrial-g; s = trial-z; ys = y.'*s;
    if ys > 1e-10*norm(y)*norm(s) && ys>0
        Hy = H*y;
        H = H + ((ys+y.'*Hy)/ys^2)*(s*s.')-(Hy*s.'+s*Hy.')/ys;
        H = 0.5*(H+H.');
    end
    oldJ = J; z = trial; J = Jtrial; g = gtrial;
    info.iterations = it;
    pg = z-min(max(z-g,lb),ub);
    if norm(pg,inf) <= opt.gradTol
        info.exitflag = 1; info.message = 'Projected gradient tolerance satisfied'; break
    end
    % A tiny cost change is diagnostic only. Continue until the gradient
    % tolerance is met, the step truly stalls, or the iteration budget ends.
    info.costStagnated = abs(oldJ-J)<=opt.functionTol*max(1,abs(oldJ));
    if norm(s,inf) <= opt.stepTol
        info.message = 'Small step; gradient tolerance not satisfied';
        break
    end
end
info.funcCount = evaluations;
info.firstorderopt = norm(z-min(max(z-g,lb),ub),inf);
end
