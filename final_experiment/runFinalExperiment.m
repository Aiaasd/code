function runFinalExperiment()
%runFinalExperiment Reproduce the final evaluation and paper figures.
% The supplied trained models and 300-sample challenge set are used.
% MATLAB R2026b, Deep Learning Toolbox, Parallel Computing Toolbox and a
% supported NVIDIA GPU are required. Data generation also uses Signal
% Processing Toolbox, Communications Toolbox and Phased Array System Toolbox.
% Results are written to stacked_results and paper_results.

evaluateStackedHoldout();
makePaperFigures_delivery();
end
