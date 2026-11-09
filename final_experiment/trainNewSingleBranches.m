function summaryTable = trainNewSingleBranches(forceRetrain)
%trainNewSingleBranches Train the MD-only model for the final comparison.

arguments
    forceRetrain (1,1) logical = false
end

cfg = isacTrainingConfig();
cfg.AccelerationCheckIterations = 20;
cfg.RandomSeed = 20271108;
cfg.DatasetRoot = fullfile(fileparts(mfilename("fullpath")),"dataset_expanded");
projectRoot = fileparts(mfilename("fullpath"));
ablationRoot = fullfile(projectRoot, "new_single_branch_results");
if ~isfolder(ablationRoot)
    mkdir(ablationRoot);
end

assert(canUseGPU, "A validated MATLAB GPU is required for ablation training.");
gpu = gpuDevice;
reset(gpu);
fprintf("Ablation training on %s with %.2f GB available memory.\n", ...
    gpu.Name, gpu.AvailableMemory/1024^3);

data = loadISACLearningData(cfg);
variants = struct(Name="md_only", DisplayName="MD only", NetworkVariant="rd_only", UseDualInput=false, ContrastiveWeight=0.05);
rows = cell(numel(variants), 1);
for variantIndex = 1:numel(variants)
    variant = variants(variantIndex);
    outputRoot = fullfile(ablationRoot, variant.Name);
    resultFile = fullfile(outputRoot, "training_results.mat");
    if isfile(resultFile) && ~forceRetrain
        fprintf("Reusing completed ablation: %s\n", variant.DisplayName);
        saved = load(resultFile, "test", "bestEpoch", "lastEpoch", "trainingMinutes");
        rows{variantIndex} = makeSummaryRow(variant.Name, variant.DisplayName, ...
            saved.test, saved.bestEpoch, saved.lastEpoch, saved.trainingMinutes);
        continue
    end
    if ~isfolder(outputRoot)
        mkdir(outputRoot);
    end
    result = trainVariant(data, cfg, variant, outputRoot);
    rows{variantIndex} = makeSummaryRow(variant.Name, variant.DisplayName, ...
        result.Test, result.BestEpoch, result.LastEpoch, result.TrainingMinutes);
end

summaryTable = vertcat(rows{:});
writetable(summaryTable, fullfile(ablationRoot, "ablation_summary.csv"));
save(fullfile(ablationRoot, "ablation_summary.mat"), "summaryTable");
fprintf("Ablation study complete: %s\n", ...
    fullfile(ablationRoot, "ablation_summary.csv"));
end

function result = trainVariant(data, cfg, variant, outputRoot)
if variant.Name == "md_only", data.RD = data.MD; end
rng(cfg.RandomSeed, "twister");
net = buildISACAblationNetwork(cfg, variant.NetworkVariant);
if variant.UseDualInput
    acceleratedLoss = dlaccelerate(@isacModelGradients);
else
    acceleratedLoss = dlaccelerate(@isacSingleInputGradients);
end

acceleratedLoss.CheckMode = "tolerance";
acceleratedLoss.CheckTolerance = 1e-3;
clearCache(acceleratedLoss);
runAccelerationCheck(net, acceleratedLoss, data, cfg, variant);
fprintf("%s dlaccelerate HitRate %.1f%%, Occupancy %.1f%%.\n", ...
    variant.DisplayName, acceleratedLoss.HitRate, acceleratedLoss.Occupancy);
assert(acceleratedLoss.HitRate >= 90, "dlaccelerate trace reuse is too low.");
acceleratedLoss.CheckMode = "none";
clearCache(acceleratedLoss);

rng(cfg.RandomSeed, "twister");
net = buildISACAblationNetwork(cfg, variant.NetworkVariant);
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
        dlTarget = dlarray(gpuArray(target), "CB");
        if variant.UseDualInput
            dlMD = dlarray(gpuArray(mdBatch), "SSCB");
            [loss, classificationLoss, contrastiveLoss, gradients, state] = ...
                dlfeval(acceleratedLoss, net, dlRD, dlMD, dlTarget, ...
                dlIdentityMask, cfg.ContrastiveTemperature, variant.ContrastiveWeight);
        else
            [loss, classificationLoss, contrastiveLoss, gradients, state] = ...
                dlfeval(acceleratedLoss, net, dlRD, dlTarget, ...
                dlIdentityMask, cfg.ContrastiveTemperature, variant.ContrastiveWeight);
        end
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

    trainEvaluation = evaluateISACAblationNetwork(net, data, ...
        data.TrainIndices, cfg, variant.UseDualInput);
    validationEvaluation = evaluateISACAblationNetwork(net, data, ...
        data.ValidationIndices, cfg, variant.UseDualInput);
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

    fprintf("%-19s epoch %3d/%d | loss %.4f | train %.1f%% | val %.1f%% | %.1f min\n", ...
        variant.Name, epoch, cfg.MaxEpochs, history.TrainingLoss(epoch), ...
        100*trainEvaluation.Accuracy, 100*validationEvaluation.Accuracy, ...
        toc(trainingTimer)/60);

    if epoch >= cfg.MinimumEpochs && epochsWithoutImprovement >= cfg.ValidationPatience
        fprintf("%s early stopping at epoch %d; best epoch %d.\n", ...
            variant.DisplayName, epoch, bestEpoch);
        break
    end
end

lastEpoch = epoch;
fields = fieldnames(history);
for fieldIndex = 1:numel(fields)
    history.(fields{fieldIndex}) = history.(fields{fieldIndex})(1:lastEpoch);
end
test = evaluateISACAblationNetwork(bestNet, data, data.TestIndices, ...
    cfg, variant.UseDualInput);
trainingMinutes = toc(trainingTimer)/60;
classNames = cfg.ClassNames;
trainingConfig = cfg;
trainingConfig.ContrastiveWeight = variant.ContrastiveWeight;
variantName = variant.Name;
save(fullfile(outputRoot, "best_model.mat"), "bestNet", "classNames", ...
    "trainingConfig", "variantName", "bestEpoch", ...
    "bestValidationAccuracy", "-v7.3");
save(fullfile(outputRoot, "training_results.mat"), "history", "test", ...
    "cfg", "variant", "bestEpoch", "bestValidationAccuracy", ...
    "lastEpoch", "trainingMinutes", "-v7.3");

historyTable = table((1:lastEpoch).', history.TrainingLoss, ...
    history.ClassificationLoss, history.ContrastiveLoss, ...
    history.TrainingAccuracy, history.ValidationAccuracy, history.LearnRate, ...
    VariableNames=["epoch" "total_loss" "classification_loss" ...
    "contrastive_loss" "training_accuracy" "validation_accuracy" "learn_rate"]);
writetable(historyTable, fullfile(outputRoot, "training_history.csv"));
writematrix(test.ConfusionMatrix, fullfile(outputRoot, "confusion_matrix.csv"));

testIndices = test.Indices;
predictions = table(data.SampleID(testIndices), ...
    cfg.ClassNames(test.Truth).', cfg.ClassNames(test.Predicted).', ...
    data.SNRDB(testIndices), data.ClutterDB(testIndices), ...
    test.Scores(1, :).', test.Scores(2, :).', test.Scores(3, :).', ...
    VariableNames=["sample_id" "true_label" "predicted_label" ...
    "snr_db" "clutter_to_target_db" ...
    "bird_probability" "drone_probability" "vehicle_probability"]);
writetable(predictions, fullfile(outputRoot, "test_predictions.csv"));
writeRobustnessTables(test, data, outputRoot);

result.Test = test;
result.BestEpoch = bestEpoch;
result.LastEpoch = lastEpoch;
result.TrainingMinutes = trainingMinutes;
end

function writeRobustnessTables(test, data, outputRoot)
testIndices = test.Indices;
snrLevels = unique(data.SNRDB(testIndices)).';
snrAccuracy = zeros(size(snrLevels));
for index = 1:numel(snrLevels)
    rows = data.SNRDB(testIndices) == snrLevels(index);
    snrAccuracy(index) = mean(test.Predicted(rows) == test.Truth(rows));
end
writetable(table(snrLevels.', snrAccuracy.', ...
    VariableNames=["snr_db" "accuracy"]), ...
    fullfile(outputRoot, "accuracy_by_snr.csv"));

clutterEdges = [-10 -5 0 5.0001];
clutterCenters = [-7.5 -2.5 2.5];
clutterAccuracy = zeros(1, 3);
for index = 1:3
    rows = data.ClutterDB(testIndices) >= clutterEdges(index) & ...
        data.ClutterDB(testIndices) < clutterEdges(index+1);
    clutterAccuracy(index) = mean(test.Predicted(rows) == test.Truth(rows));
end
writetable(table(clutterCenters.', clutterAccuracy.', ...
    VariableNames=["clutter_bin_center_db" "accuracy"]), ...
    fullfile(outputRoot, "accuracy_by_clutter.csv"));
end

function runAccelerationCheck(net, acceleratedLoss, data, cfg, variant)
batches = balancedBatches(data.Labels(data.TrainIndices), data.TrainIndices, cfg);
dlIdentityMask = dlarray(gpuArray(eye(cfg.MiniBatchSize, "single")));
trailingAverage = [];
trailingAverageSquared = [];
for iteration = 1:cfg.AccelerationCheckIterations
    batchIndex = mod(iteration-1, size(batches, 2))+1;
    indices = batches(:, batchIndex);
    [rdBatch, mdBatch] = prepareBatch(data, indices, true);
    target = oneHotTargets(data.Labels(indices), numel(cfg.ClassNames));
    dlRD = dlarray(gpuArray(rdBatch), "SSCB");
    dlTarget = dlarray(gpuArray(target), "CB");
    if variant.UseDualInput
        dlMD = dlarray(gpuArray(mdBatch), "SSCB");
        [~, ~, ~, gradients, state] = dlfeval(acceleratedLoss, net, ...
            dlRD, dlMD, dlTarget, dlIdentityMask, ...
            cfg.ContrastiveTemperature, variant.ContrastiveWeight);
    else
        [~, ~, ~, gradients, state] = dlfeval(acceleratedLoss, net, ...
            dlRD, dlTarget, dlIdentityMask, ...
            cfg.ContrastiveTemperature, variant.ContrastiveWeight);
    end
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
    classIndices{classIndex} = classIndices{classIndex}( ...
        randperm(numel(classIndices{classIndex})));
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

function row = makeSummaryRow(variantID, displayName, test, ...
    bestEpoch, completedEpochs, trainingMinutes)
row = table(string(variantID), string(displayName), test.Accuracy, ...
    test.MacroF1, test.Precision(1), test.Recall(1), test.F1(1), ...
    test.Precision(2), test.Recall(2), test.F1(2), ...
    test.Precision(3), test.Recall(3), test.F1(3), ...
    bestEpoch, completedEpochs, trainingMinutes, ...
    VariableNames=["variant_id" "display_name" "accuracy" "macro_f1" ...
    "bird_precision" "bird_recall" "bird_f1" ...
    "drone_precision" "drone_recall" "drone_f1" ...
    "vehicle_precision" "vehicle_recall" "vehicle_f1" ...
    "best_epoch" "completed_epochs" "training_minutes"]);
end
