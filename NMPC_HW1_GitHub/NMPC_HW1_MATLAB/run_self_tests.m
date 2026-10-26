function run_self_tests()
%RUN_SELF_TESTS No toolbox required. Run before the first closed-loop case.
here=fileparts(mfilename('fullpath')); addpath(here);
rng(42); cfg=nmpc_config(); p=cfg.p;
v=14.5; T=equilibrium_torque(v,p); x=[0;0;v;0;T];
xn=vehicle_step(x,[T;0],p,cfg.dt);
assert(norm(xn-[v*cfg.dt;0;v;0;T],inf)<1e-10,'Equilibrium model test failed');
cfg.Tsim=15; ref=make_reference(cfg);
assert(numel(ref.t)==round(cfg.Tsim/cfg.dt)+cfg.Np+1);
assert(abs(ref.Y(3,1)-14)<1e-12 && abs(ref.Y(3,21)-15)<1e-12);
% Compare BOTH physical Jacobians against central finite differences.
x=[2;-3;13;0.4;T]; u=[400;0.13]; h=1e-5;
[~,A,B]=vehicle_step(x,u,p,cfg.dt);
An=zeros(5); Bn=zeros(5,2);
for j=1:5
    d=zeros(5,1); d(j)=h;
    An(:,j)=(vehicle_step(x+d,u,p,cfg.dt)-vehicle_step(x-d,u,p,cfg.dt))/(2*h);
end
for j=1:2
    d=zeros(2,1); d(j)=h;
    Bn(:,j)=(vehicle_step(x,u+d,p,cfg.dt)-vehicle_step(x,u-d,p,cfg.dt))/(2*h);
end
assert(norm(A-An,'fro')/max(1,norm(A,'fro'))<1e-6,'State Jacobian failed');
assert(norm(B-Bn,'fro')/max(1,norm(B,'fro'))<1e-6,'Input Jacobian failed');
% Full-horizon gradient includes normalized inputs, increments and terminal Q+P.
k=100; Yr=ref.Y(:,k:k+cfg.Np);
Zr=[equilibrium_torque(Yr(3,2:end),p)/cfg.inputScale(1);zeros(1,cfg.Np)];
z=0.2*randn(2*cfg.Np,1); prev=[0.1;-0.03];
fun=@(q)nmpc_objective(q,x,prev,Yr,Zr,cfg);
[~,g]=fun(z); gn=zeros(size(z)); h=1e-6;
for j=1:numel(z)
    d=zeros(size(z)); d(j)=h;
    gn(j)=(fun(z+d)-fun(z-d))/(2*h);
end
relativeGradientError=norm(g-gn)/max(1,norm(gn));
assert(relativeGradientError<1e-5,'Objective gradient failed');
% Independent one-step cost checks the Q+P convention, not just derivatives.
one=cfg; one.Np=1; zo=[0.15;0.08]; zp=[0.1;-0.03];
yr=Yr(:,1:2); zr=Zr(:,1);
next=vehicle_step(x,one.inputScale.*zo,p,one.dt);
e=next(1:3)-yr(:,2); du=zo-zp; eu=zo-zr;
manualCost=e.'*(one.Q+one.P)*e+eu.'*one.R*eu+du.'*one.S*du;
assert(abs(nmpc_objective(zo,x,zp,yr,zr,one)-manualCost)<1e-9,'Terminal-cost indexing failed');
xt=x; xt(4)=4*pi+0.1; xtn=vehicle_step(xt,u,p,cfg.dt);
assert(xtn(4)>4*pi,'Heading was incorrectly wrapped');
base=vehicle_step(x,[0;0],p,cfg.dt);
more=vehicle_step(x,[1000;0],p,cfg.dt);
assert(abs(base(3)-more(3))<1e-12 && more(5)>base(5),'Actuator lag indexing failed');
% Known box-constrained minimum: unconstrained [2;-0.5], constrained [1;-0.5].
[zq,~,info]=box_bfgs(@quadratic,[0;0],[-1;-1],[1;1],cfg);
assert(norm(zq-[1;-0.5],inf)<1e-5 && info.exitflag>0,'Box optimizer failed');
warm=zeros(2,cfg.Np);
[~,s]=choose_command(warm,2,0,warm,1,cfg); assert(s==3);
[~,s]=choose_command(warm,0.5,0,warm,1,cfg); assert(s==2);
[~,s]=choose_command(warm,0.5,1,warm,1,cfg); assert(s==1);
bad=warm; bad(1)=NaN;
[~,s]=choose_command(bad,0,1,warm,1,cfg); assert(s==3);
bad=warm; bad(1)=1+1e-8;
[~,s]=choose_command(bad,0,1,warm,1,cfg); assert(s==3);
% Initial position error has 0.4 m radial magnitude; it is not averaged by axis.
r=struct('cfg',cfg,'X',[0.4 0;0 0;14 14;0 0;T T], ...
    'U',[T;0],'Z',[0.1;0],'Zref',[0.1;0], ...
    'initialNormalizedInput',[0.1;0],'t',[0 cfg.dt]);
r.ref=struct('Y',[0 0;0 0;14 14],'yawRate',[0 0]);
r.log=struct('time',1,'status',1);
m=compute_metrics(r);
assert(abs(m.full.positionRMSE_m-sqrt(0.4^2/2))<1e-12,'Position metric failed');
fprintf('PASS: model, reference indexing, Jacobians, objective gradient, box solver, fallback and metrics.\n');
fprintf('Objective-gradient relative error = %.3g\n',relativeGradientError);
end
function [J,g]=quadratic(z)
e=z-[2;-0.5]; J=e.'*e; g=2*e;
end
