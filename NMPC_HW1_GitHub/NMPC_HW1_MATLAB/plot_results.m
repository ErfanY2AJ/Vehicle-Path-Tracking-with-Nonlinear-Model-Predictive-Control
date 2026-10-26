function plot_results(r,folder)
%PLOT_RESULTS Static plots; no animation or growing plot-history buffers.
cfg=r.cfg; K=size(r.U,2); t=r.t; tu=t(1:end-1);
visible='off'; if cfg.showFigures, visible='on'; end
f=figure('Name','NMPC - tracking','Color','w','Visible',visible);
tiledlayout(2,2,'TileSpacing','compact');
nexttile; plot(r.ref.X(1,1:K+1),r.ref.X(2,1:K+1),'k--','LineWidth',1.2);
hold on; plot(r.X(1,:),r.X(2,:),'b','LineWidth',1.1); axis equal; grid on;
xlabel('x (m)'); ylabel('y (m)'); legend('Reference','Actual','Location','best');
title(sprintf('Path: alpha = %.2f',cfg.alpha));
nexttile; plot(t,r.ref.Y(3,1:K+1),'k--',t,r.X(3,:),'b'); grid on;
xlabel('Time (s)'); ylabel('Speed (m/s)'); legend('Reference','Actual');
e=r.X(1:3,:)-r.ref.Y(:,1:K+1);
nexttile; plot(t,sqrt(sum(e(1:2,:).^2,1))); grid on;
xlabel('Time (s)'); ylabel('Position error (m)');
nexttile; plot(t,e(3,:)); grid on; xlabel('Time (s)'); ylabel('Speed error (m/s)');
sgtitle(r.ref.label,'Interpreter','none');
finish(f,'tracking',cfg,folder);
f=figure('Name','NMPC - inputs','Color','w','Visible',visible);
tiledlayout(3,1,'TileSpacing','compact');
nexttile; stairs(tu,r.U(2,:)); hold on; yline(cfg.inputScale(2),'r--'); yline(-cfg.inputScale(2),'r--');
grid on; ylabel('Steering (rad)');
nexttile; stairs(tu,r.U(1,:)); hold on; plot(t,r.X(5,:));
yline(cfg.inputScale(1),'r--'); yline(-cfg.inputScale(1),'r--'); grid on; ylabel('Torque (N m)');
legend('Commanded','Actuator state','Location','best');
nexttile; refAy=r.ref.Y(3,1:K+1).*r.ref.yawRate(1:K+1);
plot(t,refAy); hold on; yline(3,'r--'); yline(-3,'r--'); grid on;
xlabel('Time (s)'); ylabel('Reference a_y (m/s^2)'); title('3 m/s^2 is indicative, not an enforced bound');
finish(f,'inputs',cfg,folder);
f=figure('Name','NMPC - solver diagnostics','Color','w','Visible',visible);
tiledlayout(3,1,'TileSpacing','compact');
nexttile; plot(tu,r.log.time); grid on; ylabel('Solve time (s)');
nexttile; plot(tu,r.log.firstorderopt); set(gca,'YScale','log'); grid on;
ylabel('Optimality measure');
nexttile; stairs(tu,r.log.status); ylim([0.7 3.3]); yticks([1 2 3]);
yticklabels({'Converged','Feasible / not converged','Fallback'});
grid on; xlabel('Time (s)');
finish(f,'solver',cfg,folder);
end
function finish(f,name,cfg,folder)
if cfg.saveFigures, exportgraphics(f,fullfile(folder,[name '.png']),'Resolution',160); end
if ~cfg.showFigures, close(f); end
end
