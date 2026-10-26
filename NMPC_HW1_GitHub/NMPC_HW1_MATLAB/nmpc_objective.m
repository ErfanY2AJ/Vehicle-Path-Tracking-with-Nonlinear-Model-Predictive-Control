function [J,g,X] = nmpc_objective(z,x0,zprev,Yref,Zref,cfg)
%NMPC_OBJECTIVE Single shooting, Eq. (4), and an analytic adjoint gradient.
% Z(:,j) represents u_bar_(j-1|t); X(:,j+1) represents x_(j|t).
% Yref includes the current state and all Np future outputs.
% Zref(:,j) uses Teq(vref_(t+j)), i.e. NEXT reference speed for input j-1.
N = cfg.Np;
Z = reshape(z,2,N);
X = zeros(5,N+1); X(:,1) = x0;
wantGradient = nargout >= 2;
if wantGradient
    AA = zeros(5,5,N); BB = zeros(5,2,N);
end
J = 0;
for j = 1:N
    u = cfg.inputScale.*Z(:,j);
    if wantGradient
        [X(:,j+1),AA(:,:,j),Bphysical] = vehicle_step(X(:,j),u,cfg.p,cfg.dt);
        BB(:,:,j) = Bphysical*diag(cfg.inputScale);
    else
        X(:,j+1) = vehicle_step(X(:,j),u,cfg.p,cfg.dt);
    end
    e = X(1:3,j+1)-Yref(:,j+1);
    J = J + e.'*cfg.Q*e;
end
e = X(1:3,end)-Yref(:,end);
J = J + e.'*cfg.P*e; % Q at j=Np PLUS terminal P: both are deliberate.
E = Z-Zref;
D = Z-[zprev,Z(:,1:end-1)];
J = J + sum(sum(E.*(cfg.R*E))) + sum(sum(D.*(cfg.S*D)));
if wantGradient
    G = 2*cfg.R*E + 2*cfg.S*D;
    G(:,1:end-1) = G(:,1:end-1)-2*cfg.S*D(:,2:end);
    lambda = [2*(cfg.Q+cfg.P)*e;0;0];
    for j = N:-1:1
        G(:,j) = G(:,j) + BB(:,:,j).'*lambda;
        if j > 1
            eprev = X(1:3,j)-Yref(:,j);
            lambda = AA(:,:,j).'*lambda + [2*cfg.Q*eprev;0;0];
        end
    end
    g = G(:);
end
end
