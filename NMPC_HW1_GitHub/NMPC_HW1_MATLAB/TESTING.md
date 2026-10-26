# Validation scope

## Reported native MATLAB results

The English README includes MATLAB outputs supplied after the original
implementation was delivered: baseline, challenging-path, mass-mismatch,
step-tolerance, and automatic-tuning experiments. These are reported
executions on the experimenter's machine. They are distinct from the
independent NumPy checks below. No new native MATLAB execution was
performed while preparing or revising this report. The transcribed
summary files in `reported_results/` do not contain original raw MAT
trajectories. The README labels this provenance explicitly.

## Added automatic tuner

The automatic tuner and its MATLAB helper tests have NOT been executed in
MATLAB in this environment. Run `run_tuning_tests`, then the small `quick`
mode locally. The fixed outer-score formula was independently checked with
synthetic numerical cases (zero errors, larger errors, time overruns and
non-convergence). Those checks are not native MATLAB execution or evidence
that a tuned controller improves real closed-loop performance. The core
controller checks below predate the outer tuner and do not validate its
full search/validation workflow.

## Native MATLAB execution

MATLAB and Octave were unavailable in the authoring environment. The .m files were not executed in MATLAB by the report author. In particular, MATLAB graphics, CSV/table export, the optional fmincon interface and native function dispatch remain to be checked locally. Source function names, file names and the core numerical indexing were reviewed.

Run `run_self_tests` in MATLAB before running `run_nmpc_hw1`. The supplied MATLAB tests check equilibrium motion, reference sample count/speed, analytical state and input Jacobians, the full-horizon gradient, an independent one-step terminal-cost formula, unwrapped heading, actuator delay, a known constrained quadratic minimum, fallback selection, invalid inputs, and radial position-error metrics. These local tests are supplied but were not run here.

## Independent executable numerical checks

An independent NumPy implementation of the model, objective, adjoint gradient and custom optimizer was executed. It checks the numerical design; it is not a MATLAB compatibility test. Raw results and the reproducible verification script are in `validation/`. That script requires NumPy and is optional; it is not used by the MATLAB application.

Analytical Jacobians were compared against central finite differences:

| Check | Relative error |
|---|---:|
| State Jacobian | 5.34e-11 |
| Physical-input Jacobian | 1.23e-09 |

| Horizon | dt (s) | Objective-gradient relative error |
|---:|---:|---:|
| 1 | 0.1 | 3.41e-06 |
| 10 | 0.05 | 6.21e-08 |
| 20 | 0.1 | 1.01e-08 |
| 30 | 0.2 | 2.05e-09 |

The independent port completed these closed-loop tests with finite states, positive actual speed and all normalized applied inputs inside [-1,1]:

| Duration (s) | Practice alpha | Actual mass factor | Converged | Feasible/not converged | Fallback |
|---:|---:|---:|---:|---:|---:|
| 15 | 0.3 | 1 | 150 | 0 | 0 |
| 100 | 0.3 | 1 | 992 | 8 | 0 |
| 100 | 1 | 1 | 996 | 4 | 0 |
| 100 | 0.3 | 1.1 | 916 | 84 | 0 |

The counters intentionally retain non-converged steps; no performance or global-optimality guarantee is implied. These are tests of the explicitly defined PRACTICE reference. They are not results for the unavailable original assignment reference. The six sensitivity variants and optional fmincon solver have not been executed here. Any elapsed times in the JSON belong to the Python verification process, not MATLAB.

## Interpretation limits

The kinematic lateral model omits tire-slip dynamics. Tracking success is not a physical-feasibility or safety certificate. The 3 m/s^2 line on the plot is an indicative reference only.

## Primary documentation

The optional fmincon call uses the documented SQP algorithm, bound arguments and SpecifyObjectiveGradient interface: https://www.mathworks.com/help/optim/ug/fmincon.html
