function createISACTrainingFigures(history, test, data, cfg)
%createISACTrainingFigures Export compact figures and result tables.

figureRoot = fullfile(cfg.OutputRoot, "figures");
if ~isfolder(figureRoot)
    mkdir(figureRoot);
end

epoch = (1:numel(history.TrainingLoss)).';
figureHandle = figure(Visible="off", Position=[100 100 1100 430]);
tiledlayout(1, 2, TileSpacing="compact", Padding="compact");
nexttile;
plot(epoch, history.TrainingLoss, LineWidth=1.5); hold on;
plot(epoch, history.ClassificationLoss, LineWidth=1.2);
plot(epoch, history.ContrastiveLoss, LineWidth=1.2);
grid on; xlabel("Epoch"); ylabel("Loss");
legend("Total", "Classification", "Supervised contrastive", Location="northeast");
title("Training losses");
nexttile;
plot(epoch, 100*history.ValidationAccuracy, LineWidth=1.8); hold on;
plot(epoch, 100*history.TrainingAccuracy, LineWidth=1.2);
grid on; ylim([0 100]); xlabel("Epoch"); ylabel("Accuracy (%)");
legend("Validation", "Training", Location="southeast");
title("Classification accuracy");
exportgraphics(figureHandle, fullfile(figureRoot, "training_curves.png"), Resolution=180);
close(figureHandle);

confusion = test.ConfusionMatrix;
rowNormalized = confusion./max(sum(confusion, 2), 1);
figureHandle = figure(Visible="off", Position=[100 100 620 520]);
imagesc(100*rowNormalized, [0 100]); axis image; colormap(turbo); colorbar;
xticks(1:numel(cfg.ClassNames)); yticks(1:numel(cfg.ClassNames));
xticklabels(upper(cfg.ClassNames)); yticklabels(upper(cfg.ClassNames));
xlabel("Predicted class"); ylabel("True class"); title("Test confusion matrix (%)");
for row = 1:size(confusion, 1)
    for column = 1:size(confusion, 2)
        text(column, row, sprintf("%.1f%%\n(%d)", ...
            100*rowNormalized(row, column), confusion(row, column)), ...
            HorizontalAlignment="center", FontWeight="bold", Color="white");
    end
end
exportgraphics(figureHandle, fullfile(figureRoot, "confusion_matrix.png"), Resolution=180);
close(figureHandle);

testIndices = test.Indices;
snrLevels = unique(data.SNRDB(testIndices)).';
snrAccuracy = zeros(size(snrLevels));
for index = 1:numel(snrLevels)
    rows = data.SNRDB(testIndices) == snrLevels(index);
    snrAccuracy(index) = mean(test.Predicted(rows) == test.Truth(rows));
end
figureHandle = figure(Visible="off", Position=[100 100 620 430]);
plot(snrLevels, 100*snrAccuracy, "-o", LineWidth=2, MarkerSize=7);
grid on; ylim([0 100]); xticks(snrLevels);
xlabel("SNR (dB)"); ylabel("Accuracy (%)"); title("Recognition accuracy versus SNR");
exportgraphics(figureHandle, fullfile(figureRoot, "accuracy_vs_snr.png"), Resolution=180);
close(figureHandle);

clutterEdges = [-10 -5 0 5.0001];
clutterCenters = [-7.5 -2.5 2.5];
clutterAccuracy = zeros(1, 3);
for index = 1:3
    rows = data.ClutterDB(testIndices) >= clutterEdges(index) & ...
        data.ClutterDB(testIndices) < clutterEdges(index+1);
    clutterAccuracy(index) = mean(test.Predicted(rows) == test.Truth(rows));
end
figureHandle = figure(Visible="off", Position=[100 100 620 430]);
plot(clutterCenters, 100*clutterAccuracy, "-s", LineWidth=2, MarkerSize=7);
grid on; ylim([0 100]); xticks(clutterCenters);
xlabel("Clutter-to-target ratio bin center (dB)"); ylabel("Accuracy (%)");
title("Recognition accuracy versus clutter level");
exportgraphics(figureHandle, fullfile(figureRoot, "accuracy_vs_clutter.png"), Resolution=180);
close(figureHandle);

embedding = double(test.Embeddings.');
embedding = embedding-mean(embedding, 1);
[u, s, ~] = svd(embedding, "econ");
pcaCoordinates = u(:, 1:2)*s(1:2, 1:2);
figureHandle = figure(Visible="off", Position=[100 100 620 500]); hold on;
colors = lines(numel(cfg.ClassNames));
for classIndex = 1:numel(cfg.ClassNames)
    rows = test.Truth == classIndex;
    scatter(pcaCoordinates(rows, 1), pcaCoordinates(rows, 2), 32, ...
        colors(classIndex, :), "filled", DisplayName=upper(cfg.ClassNames(classIndex)));
end
grid on; xlabel("Principal component 1"); ylabel("Principal component 2");
title("Test embeddings"); legend(Location="best");
exportgraphics(figureHandle, fullfile(figureRoot, "embedding_pca.png"), Resolution=180);
close(figureHandle);

figureHandle = figure(Visible="off", Position=[100 100 1200 850]);
tiledlayout(2, 2, TileSpacing="compact", Padding="compact");
nexttile;
imagesc(100*rowNormalized, [0 100]); axis image; colormap(turbo); colorbar;
xticks(1:3); yticks(1:3); xticklabels(upper(cfg.ClassNames)); yticklabels(upper(cfg.ClassNames));
xlabel("Predicted"); ylabel("True"); title("(a) Confusion matrix (%)");
nexttile;
plot(snrLevels, 100*snrAccuracy, "-o", LineWidth=2); grid on; ylim([0 100]);
xlabel("SNR (dB)"); ylabel("Accuracy (%)"); title("(b) Low-SNR robustness");
nexttile;
plot(clutterCenters, 100*clutterAccuracy, "-s", LineWidth=2); grid on; ylim([0 100]);
xlabel("Clutter/target (dB)"); ylabel("Accuracy (%)"); title("(c) Clutter robustness");
nexttile; hold on;
for classIndex = 1:numel(cfg.ClassNames)
    rows = test.Truth == classIndex;
    scatter(pcaCoordinates(rows, 1), pcaCoordinates(rows, 2), 24, ...
        colors(classIndex, :), "filled", DisplayName=upper(cfg.ClassNames(classIndex)));
end
grid on; xlabel("PC1"); ylabel("PC2"); title("(d) Learned embeddings"); legend(Location="best");
exportgraphics(figureHandle, fullfile(figureRoot, "paper_results_composite.png"), Resolution=180);
close(figureHandle);

writetable(table(snrLevels.', snrAccuracy.', ...
    VariableNames=["snr_db" "accuracy"]), fullfile(cfg.OutputRoot, "accuracy_by_snr.csv"));
writetable(table(clutterCenters.', clutterAccuracy.', ...
    VariableNames=["clutter_bin_center_db" "accuracy"]), ...
    fullfile(cfg.OutputRoot, "accuracy_by_clutter.csv"));
end
