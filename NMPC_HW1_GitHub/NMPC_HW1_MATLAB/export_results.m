function export_results(r,folder)
%EXPORT_RESULTS Raw states, commands and per-step solver diagnostics.
K = size(r.U,2);
states = table(r.t.',r.X(1,:).',r.X(2,:).',r.X(3,:).',r.X(4,:).',r.X(5,:).', ...
    r.ref.X(1,1:K+1).',r.ref.X(2,1:K+1).',r.ref.X(3,1:K+1).',r.ref.X(4,1:K+1).', ...
    'VariableNames',{'t_s','px_m','py_m','v_mps','phi_rad','T_Nm', ...
    'px_ref_m','py_ref_m','v_ref_mps','phi_ref_rad'});
writetable(states,fullfile(folder,'states.csv'));
commands = table(r.t(1:end-1).',r.U(1,:).',r.U(2,:).', ...
    r.Z(1,:).',r.Z(2,:).','VariableNames', ...
    {'t_s','torqueCommand_Nm','steering_rad','normalizedTorque','normalizedSteering'});
writetable(commands,fullfile(folder,'commands.csv'));
solverLog=struct2table(r.log);
solverLog.t_s=r.t(1:end-1).';
writetable(solverLog,fullfile(folder,'solver_log.csv'));
fid=fopen(fullfile(folder,'run_notes.txt'),'w');
if fid<0, error('Cannot write run notes.'); end
cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'Reference: %s\nSolver: %s\nMATLAB: %s\n',r.ref.label,r.cfg.solver,version);
fprintf(fid,'Termination: %s\nCompleted: %d\n',r.termination,r.completed);
fprintf(fid,'Status 1 = solver-reported convergence, 2 = feasible/not converged, 3 = fallback.\n');
fprintf(fid,'Timing includes warm-cost evaluation, optimization and command selection.\n');
fprintf(fid,'Original physical model, objective and bounds; alternative MATLAB solvers.\n');
fprintf(fid,'Reference ay peak = %.9g m/s^2 (diagnostic, not a safety constraint).\n', ...
    r.metrics.referencePeakLateralAcceleration_mps2);
fprintf(fid,'Applied-input bound violations = %d\n',r.metrics.boundViolations);
fprintf(fid,'Non-finite applied inputs = %d\n',r.metrics.nonfiniteAppliedInputs);
end
