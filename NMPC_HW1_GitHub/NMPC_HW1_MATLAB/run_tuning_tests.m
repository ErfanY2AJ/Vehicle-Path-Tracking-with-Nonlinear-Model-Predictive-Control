function run_tuning_tests()
%RUN_TUNING_TESTS Fast helper tests; no closed-loop simulation or toolbox.
base=nmpc_config(); opt=nmpc_tuning_options('standard');
c=nmpc_tuning_decode([20,0,0,0,0],base,opt);
assert(isequal(c.Q,base.Q) && isequal(c.P,base.P) && isequal(c.R,base.R) && isequal(c.S,base.S));
c=nmpc_tuning_decode([30,1,-1,1,-1],base,opt);
assert(c.Np==30 && c.dt==base.dt && c.gradTol==base.gradTol && c.stepTol==base.stepTol);
assert(c.Q(1,1)==base.Q(1,1) && c.P(1,1)==base.P(1,1));
assert(c.Q(3,3)==4*base.Q(3,3) && c.R(1,1)==base.R(1,1)/4);
assert(c.S(1,1)==4*base.S(1,1) && c.S(2,2)==base.S(2,2)/4);
% Mock measured metrics let us check the OUTER objective independently.
m=struct('positionRMSE_m',0,'speedRMSE_mps',0,'Eu',0,'EdeltaU',0);
r=struct('completed',true,'X',zeros(5,10),'cfg',base);
r.X(3,:)=15;
r.metrics=struct('full',m,'after5s',m,'boundViolations',0,'nonfiniteAppliedInputs',0);
r.log=struct('time',0.01*ones(10,1),'status',ones(10,1));
s0=nmpc_tuning_score(r,opt);
assert(abs(s0-opt.timeWeight*(0.01/base.dt)^2)<1e-12);
r2=r; r2.metrics.after5s.positionRMSE_m=opt.positionPostScale;
assert(abs(nmpc_tuning_score(r2,opt)-s0-1)<1e-12);
r2=r; r2.cfg.Q=base.Q*100;
assert(nmpc_tuning_score(r2,opt)==s0,'Outer score must not reuse reweighted MPC J');
r2=r; r2.log.time(:)=0.2;
assert(nmpc_tuning_score(r2,opt)>s0);
r2=r; r2.log.status(:)=2;
assert(abs(nmpc_tuning_score(r2,opt)-s0-opt.nonconvergenceWeight)<1e-12);
r2=r; r2.log.status(:)=3;
assert(abs(nmpc_tuning_score(r2,opt)-s0-10)<1e-12);
r2=r; r2.completed=false; assert(isinf(nmpc_tuning_score(r2,opt)));
r2=r; r2.metrics.boundViolations=1; assert(isinf(nmpc_tuning_score(r2,opt)));
r2=r; r2.X(1)=NaN; assert(isinf(nmpc_tuning_score(r2,opt)));
r2=r; r2.X(3,1)=0; assert(isinf(nmpc_tuning_score(r2,opt)));
fprintf('PASS: baseline decoding, weight bounds, frozen solver settings, fixed outer score and invalid-run rejection.\n');
end
