function generateStackedHoldout()
% Fresh three-class holdout, 100 per class, fixed before final selection.
root=fileparts(mfilename('fullpath')); cfg=isacDatasetConfig();
cfg.RandomSeed=27000000; cfg.SNRLevelsDB=[-15 -10 -5 0 5];
cfg.SamplesPerSNR=20; cfg.SamplesPerClass=100; cfg.ChallengeMode=true;
cfg.ClutterToTargetRangeDB=[5 12]; cfg.ClutterCountRange=[12 24];
cfg.OutputRoot=fullfile(root,'stacked_holdout_raw');
if ~isfile(fullfile(cfg.OutputRoot,'manifest.csv')), generateISACDataset(cfg);end
rebuildRDdataset(cfg.OutputRoot,fullfile(root,'stacked_holdout_pooled'));
end
