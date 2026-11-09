function evaluation = evaluateISACNetwork(net, data, indices, cfg)
%evaluateISACNetwork Predict labels, scores, and embeddings in GPU batches.

numSamples = numel(indices);
numClasses = numel(cfg.ClassNames);
scores = zeros(numClasses, numSamples, "single");
embeddings = zeros(cfg.EmbeddingDimension, numSamples, "single");

for first = 1:cfg.ValidationBatchSize:numSamples
    last = min(first+cfg.ValidationBatchSize-1, numSamples);
    local = first:last;
    selected = indices(local);
    rd = dlarray(gpuArray(data.RD(:, :, :, selected)), "SSCB");
    md = dlarray(gpuArray(data.MD(:, :, :, selected)), "SSCB");
    [embedding, probability] = predict(net, rd, md, ...
        Outputs=["embedding" "class_probs"]);
    scores(:, local) = gather(extractdata(probability));
    embeddings(:, local) = gather(extractdata(embedding));
end

[~, predicted] = max(scores, [], 1);
truth = double(data.Labels(indices)).';
confusion = accumarray([truth(:) predicted(:)], 1, [numClasses numClasses]);
accuracy = mean(predicted == truth);

truePositive = diag(confusion);
precision = truePositive./max(sum(confusion, 1).', 1);
recall = truePositive./max(sum(confusion, 2), 1);
f1 = 2*precision.*recall./max(precision+recall, eps);

evaluation.Indices = indices;
evaluation.Truth = truth;
evaluation.Predicted = predicted;
evaluation.Scores = scores;
evaluation.Embeddings = embeddings;
evaluation.ConfusionMatrix = confusion;
evaluation.Accuracy = accuracy;
evaluation.Precision = precision;
evaluation.Recall = recall;
evaluation.F1 = f1;
evaluation.MacroF1 = mean(f1);
end
