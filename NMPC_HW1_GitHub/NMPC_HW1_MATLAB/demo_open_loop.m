function demo_open_loop()
%DEMO_OPEN_LOOP First learning step: model without a controller.
cfg=nmpc_config(); p=cfg.p; dt=cfg.dt; t=0:dt:10;
x=zeros(5,numel(t)); x(:,1)=[0;0;14;0;equilibrium_torque(14,p)];
for k=1:numel(t)-1
    u=[equilibrium_torque(14,p);0];
    if t(k)>=3 && t(k)<5, u(2)=0.05; end
    x(:,k+1)=vehicle_step(x(:,k),u,p,dt);
end
figure('Color','w'); tiledlayout(1,2);
nexttile; plot(x(1,:),x(2,:)); axis equal; grid on;
xlabel('x (m)'); ylabel('y (m)'); title('Steering pulse at 3-5 seconds');
nexttile; plot(t,x(3,:)); grid on; xlabel('Time (s)'); ylabel('Speed (m/s)');
title('Constant equilibrium torque');
end
