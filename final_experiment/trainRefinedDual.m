function trainRefinedDual()
% Train dual networks on the pooled RD maps with fixed random seeds.
root=fileparts(mfilename('fullpath'));
cfg=isacTrainingConfig(); cfg.DatasetRoot=fullfile(root,'dataset_rd_pooled');
for seed=[20271108 20271109 20271110]
    cfg.RandomSeed=seed;
    cfg.OutputRoot=fullfile(root,'refined_dual_candidates',sprintf('seed_%d',seed));
    if ~isfile(fullfile(cfg.OutputRoot,'training_results.mat'))
        trainISACClassifier(cfg);
    end
end
end
