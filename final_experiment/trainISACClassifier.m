function output = trainISACClassifier(cfg)
%trainISACClassifier Train the RD/MD attention model with supervised contrastive loss.

arguments
    cfg (1,1) struct = isacTrainingConfig()
end

close all;
rng(cfg.RandomSeed, "twister");
if ~isfolder(cfg.OutputRoot)
    mkdir(cfg.OutputRoot);
end
checkpointRoot = fullfile(cfg.OutputRoot, "checkpoints");
if ~isfolder(checkpointRoot)
    mkdir(checkpointRoot);
end

assert(canUseGPU, "A validated MATLAB GPU is required for formal training.");
gpu = gpuDevice;
reset(gpu);
fprintf("Training on %s with %.2f GB available memory.\n", ...
    gpu.Name, gpu.AvailableMemory/1024^3);

data = loadISACLearningData(cfg);
net = buildDualBranchISACNetwork(cfg);
acceleratedLoss = dlaccelerate(@isacModelGradients);

% Verify trace reuse and numerical equivalence using real balanced batches.
fprintf("Verifying dlaccelerate with %d weight updates.\n", cfg.AccelerationCheckIterations);
acceleratedLoss.CheckMode = "tolerance";
acceleratedLoss.CheckTolerance = 1e-3;
clearCache(acceleratedLoss);
runAccelerationCheck(net, acceleratedLoss, data, cfg);
fprintf("dlaccelerate HitRate %.1f%%, Occupancy %.1f%%.\n", ...
    acceleratedLoss.HitRate, acceleratedLoss.Occupancy);
assert(acceleratedLoss.HitRate >= 90, "dlaccelerate trace reuse is too low.");
acceleratedLoss.CheckMode = "none";
clearCache(acceleratedLoss);

% Reinitialize after the acceleration check so formal training is clean.
net = buildDualBranchISACNetwork(cfg);
trailingAverage = [];
trailingAverageSquared = [];
iteration = 0;
numBatches = numel(data.TrainIndices)/cfg.MiniBatchSize;
dlIdentityMask = dlarray(gpuArray(eye(cfg.MiniBatchSize, "single")));

history.TrainingLoss = nan(cfg.MaxEpochs, 1);
history.ClassificationLoss = nan(cfg.MaxEpochs, 1);
history.ContrastiveLoss = nan(cfg.MaxEpochs, 1);
history.TrainingAccuracy = nan(cfg.MaxEpochs, 1);
history.ValidationAccuracy = nan(cfg.MaxEpochs, 1);
history.LearnRate = nan(cfg.MaxEpochs, 1);

bestValidationAccuracy = -inf;
bestEpoch = 0;
bestNet = net;
epochsWithoutImprovement = 0;
trainingTimer = tic;

for epoch = 1:cfg.MaxEpochs
    learningRate = epochLearningRate(epoch, cfg);
    batches = balancedBatches(data.Labels(data.TrainIndices), ...
        data.TrainIndices, cfg);
    epochLoss = zeros(numBatches, 1);
    epochClassification = zeros(numBatches, 1);
    epochContrastive = zeros(numBatches, 1);

    for batchIndex = 1:numBatches
        iteration = iteration+1;
        indices = batches(:, batchIndex);
        [rdBatch, mdBatch] = prepareBatch(data, indices, cfg.EnableAugmentation);
        target = oneHotTargets(data.Labels(indices), numel(cfg.ClassNames));
        dlRD = dlarray(gpuArray(rdBatch), "SSCB");
        dlMD = dlarray(gpuArray(mdBatch), "SSCB");
        dlTarget = dlarray(gpuArray(target), "CB");

        [loss, classificationLoss, contrastiveLoss, gradients, state] = ...
            dlfeval(acceleratedLoss, net, dlRD, dlMD, dlTarget, ...
            dlIdentityMask, cfg.ContrastiveTemperature, cfg.ContrastiveWeight);
        net.State = state;
        gradients = dlupdate(@(gradient, parameter) ...
            gradient+cfg.WeightDecay*parameter, gradients, net.Learnables);
        [net, trailingAverage, trailingAverageSquared] = adamupdate( ...
            net, gradients, trailingAverage, trailingAverageSquared, ...
            iteration, learningRate);

        epochLoss(batchIndex) = double(gather(extractdata(loss)));
        epochClassification(batchIndex) = double(gather(extractdata(classificationLoss)));
        epochContrastive(batchIndex) = double(gather(extractdata(contrastiveLoss)));
    end

    trainEvaluation = evaluateISACNetwork(net, data, data.TrainIndices, cfg);
    validationEvaluation = evaluateISACNetwork(net, data, data.ValidationIndices, cfg);
    history.TrainingLoss(epoch) = mean(epochLoss);
    history.ClassificationLoss(epoch) = mean(epochClassification);
    history.ContrastiveLoss(epoch) = mean(epochContrastive);
    history.TrainingAccuracy(epoch) = trainEvaluation.Accuracy;
    history.ValidationAccuracy(epoch) = validationEvaluation.Accuracy;
    history.LearnRate(epoch) = learningRate;

    if validationEvaluation.Accuracy > bestValidationAccuracy
        bestValidationAccuracy = validationEvaluation.Accuracy;
        bestEpoch = epoch;
        bestNet = net;
        epochsWithoutImprovement = 0;
    else
        epochsWithoutImprovement = epochsWithoutImprovement+1;
    end

    fprintf("Epoch %3d/%d | loss %.4f (CE %.4f, SupCon %.4f) | train %.1f%% | val %.1f%% | %.1f min\n", ...
        epoch, cfg.MaxEpochs, history.TrainingLoss(epoch), ...
        history.ClassificationLoss(epoch), history.ContrastiveLoss(epoch), ...
        100*trainEvaluation.Accuracy, 100*validationEvaluation.Accuracy, ...
        toc(trainingTimer)/60);

    if mod(epoch, cfg.CheckpointFrequency) == 0
        checkpoint = fullfile(checkpointRoot, sprintf("checkpoint_epoch_%03d.mat", epoch));
        save(checkpoint, "net", "history", "cfg", "epoch", "-v7.3");
    end
    if epoch >= cfg.MinimumEpochs && epochsWithoutImprovement >= cfg.ValidationPatience
        fprintf("Early stopping at epoch %d. Best validation epoch: %d.\n", epoch, bestEpoch);
        break
    end
end

lastEpoch = epoch;
fields = fieldnames(history);
for fieldIndex = 1:numel(fields)
    history.(fields{fieldIndex}) = history.(fields{fieldIndex})(1:lastEpoch);
end

validation = evaluateISACNetwork(bestNet, data, data.ValidationIndices, cfg);
test = evaluateISACNetwork(bestNet, data, data.TestIndices, cfg);
classNames = cfg.ClassNames;
trainingConfig = cfg;
save(fullfile(cfg.OutputRoot, "best_model.mat"), ...
    "bestNet", "classNames", "trainingConfig", "bestEpoch", ...
    "bestValidationAccuracy", "-v7.3");
save(fullfile(cfg.OutputRoot, "training_results.mat"), ...
    "history", "validation", "test", "cfg", "bestEpoch", "-v7.3");

historyTable = table((1:lastEpoch).', history.TrainingLoss, ...
    history.ClassificationLoss, history.ContrastiveLoss, ...
    history.TrainingAccuracy, history.ValidationAccuracy, history.LearnRate, ...
    VariableNames=["epoch" "total_loss" "classification_loss" ...
    "contrastive_loss" "training_accuracy" "validation_accuracy" "learn_rate"]);
writetable(historyTable, fullfile(cfg.OutputRoot, "training_history.csv"));

testIndices = test.Indices;
predictionTable = table(data.SampleID(testIndices), ...
    cfg.ClassNames(test.Truth).', cfg.ClassNames(test.Predicted).', ...
    data.SNRDB(testIndices), data.ClutterDB(testIndices), ...
    test.Scores(1, :).', test.Scores(2, :).', test.Scores(3, :).', ...
    VariableNames=["sample_id" "true_label" "predicted_label" "snr_db" ...
    "clutter_to_target_db" "bird_probability" "drone_probability" "vehicle_probability"]);
writetable(predictionTable, fullfile(cfg.OutputRoot, "test_predictions.csv"));

summary = table(test.Accuracy, test.MacroF1, bestValidationAccuracy, ...
    bestEpoch, lastEpoch, toc(trainingTimer)/60, ...
    VariableNames=["test_accuracy" "macro_f1" "best_validation_accuracy" ...
    "best_epoch" "completed_epochs" "training_minutes"]);
writetable(summary, fullfile(cfg.OutputRoot, "performance_summary.csv"));
writematrix(test.ConfusionMatrix, fullfile(cfg.OutputRoot, "confusion_matrix.csv"));
createISACTrainingFigures(history, test, data, cfg);

fprintf("Formal training complete. Test accuracy %.2f%%, macro-F1 %.4f.\n", ...
    100*test.Accuracy, test.MacroF1);
fprintf("Best model: %s\n", fullfile(cfg.OutputRoot, "best_model.mat"));

output.BestModelFile = fullfile(cfg.OutputRoot, "best_model.mat");
output.TestAccuracy = test.Accuracy;
output.MacroF1 = test.MacroF1;
output.BestEpoch = bestEpoch;
output.CompletedEpochs = lastEpoch;
output.TrainingMinutes = toc(trainingTimer)/60;
end

function [net, trailingAverage, trailingAverageSquared] = ...
    runAccelerationCheck(net, acceleratedLoss, data, cfg)
trailingAverage = [];
trailingAverageSquared = [];
batches = balancedBatches(data.Labels(data.TrainIndices), data.TrainIndices, cfg);
dlIdentityMask = dlarray(gpuArray(eye(cfg.MiniBatchSize, "single")));
for iteration = 1:cfg.AccelerationCheckIterations
    batchIndex = mod(iteration-1, size(batches, 2))+1;
    indices = batches(:, batchIndex);
    [rdBatch, mdBatch] = prepareBatch(data, indices, true);
    target = oneHotTargets(data.Labels(indices), numel(cfg.ClassNames));
    dlRD = dlarray(gpuArray(rdBatch), "SSCB");
    dlMD = dlarray(gpuArray(mdBatch), "SSCB");
    dlTarget = dlarray(gpuArray(target), "CB");
    [~, ~, ~, gradients, state] = dlfeval(acceleratedLoss, net, ...
        dlRD, dlMD, dlTarget, dlIdentityMask, ...
        cfg.ContrastiveTemperature, cfg.ContrastiveWeight);
    net.State = state;
    [net, trailingAverage, trailingAverageSquared] = adamupdate(net, gradients, ...
        trailingAverage, trailingAverageSquared, iteration, cfg.InitialLearnRate);
end
end

function batches = balancedBatches(localLabels, globalIndices, cfg)
numClasses = numel(cfg.ClassNames);
perClass = cfg.SamplesPerClassPerBatch;
numBatches = numel(globalIndices)/cfg.MiniBatchSize;
batches = zeros(cfg.MiniBatchSize, numBatches);
classIndices = cell(1, numClasses);
for classIndex = 1:numClasses
    classIndices{classIndex} = globalIndices(localLabels == classIndex);
    classIndices{classIndex} = classIndices{classIndex}(randperm(numel(classIndices{classIndex})));
end
for batchIndex = 1:numBatches
    batch = zeros(cfg.MiniBatchSize, 1);
    cursor = 1;
    for classIndex = 1:numClasses
        source = (batchIndex-1)*perClass+(1:perClass);
        batch(cursor:cursor+perClass-1) = classIndices{classIndex}(source);
        cursor = cursor+perClass;
    end
    batches(:, batchIndex) = batch(randperm(cfg.MiniBatchSize));
end
end

function [rdBatch, mdBatch] = prepareBatch(data, indices, augment)
rdBatch = data.RD(:, :, :, indices);
mdBatch = data.MD(:, :, :, indices);
if ~augment
    return
end
for sampleIndex = 1:numel(indices)
    rdBatch(:, :, 1, sampleIndex) = circshift(rdBatch(:, :, 1, sampleIndex), ...
        [randi([-2 2]) randi([-4 4])]);
    mdBatch(:, :, 1, sampleIndex) = circshift(mdBatch(:, :, 1, sampleIndex), ...
        [randi([-2 2]) randi([-4 4])]);
end
rdBatch = min(max(rdBatch+0.01*randn(size(rdBatch), "single"), 0), 1);
mdBatch = min(max(mdBatch+0.01*randn(size(mdBatch), "single"), 0), 1);
end

function targets = oneHotTargets(labels, numClasses)
targets = single((1:numClasses).' == double(labels(:).'));
end

function learningRate = epochLearningRate(epoch, cfg)
if epoch <= 60
    learningRate = cfg.InitialLearnRate;
elseif epoch <= 85
    learningRate = 0.3*cfg.InitialLearnRate;
else
    learningRate = 0.1*cfg.InitialLearnRate;
end
end
