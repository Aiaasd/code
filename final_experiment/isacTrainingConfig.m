function cfg = isacTrainingConfig()
%isacTrainingConfig Training settings for the dual-branch ISAC classifier.

projectRoot = fileparts(mfilename("fullpath"));
cfg.DatasetRoot = fullfile(projectRoot, "dataset");
cfg.OutputRoot = fullfile(projectRoot, "training_results");
cfg.ClassNames = ["bird" "drone" "vehicle"];
cfg.RandomSeed = 20261106;

cfg.ImageSize = [128 128 1];
cfg.EmbeddingDimension = 128;
cfg.MiniBatchSize = 30;
cfg.SamplesPerClassPerBatch = 10;
cfg.MaxEpochs = 100;
cfg.InitialLearnRate = 5e-4;
cfg.ContrastiveTemperature = 0.07;
cfg.ContrastiveWeight = 0.05;
cfg.WeightDecay = 1e-4;
cfg.ValidationPatience = 20;
cfg.MinimumEpochs = 40;
cfg.ValidationBatchSize = 45;
cfg.CheckpointFrequency = 10;
cfg.UseGPU = true;
cfg.EnableAugmentation = true;
cfg.AccelerationCheckIterations = 20;
end
