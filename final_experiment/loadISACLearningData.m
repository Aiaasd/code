function data = loadISACLearningData(cfg)
%loadISACLearningData Load paired RD/MD tensors and metadata into memory.

manifestFile = fullfile(cfg.DatasetRoot, "manifest.csv");
if ~isfile(manifestFile)
    error("Dataset manifest not found: %s", manifestFile);
end
manifest = readtable(manifestFile, TextType="string");
numSamples = height(manifest);

rd = zeros([cfg.ImageSize numSamples], "single");
md = zeros([cfg.ImageSize numSamples], "single");
labels = zeros(numSamples, 1, "uint8");
clutterDB = zeros(numSamples, 1);

fprintf("Loading %d paired RD/MD samples into memory.\n", numSamples);
for sampleIndex = 1:numSamples
    sample = load(fullfile(cfg.DatasetRoot, strrep(manifest.mat_path(sampleIndex), "\", filesep)), ...
        "rdMap", "mdMap", "metadata");
    rd(:, :, 1, sampleIndex) = single(sample.rdMap);
    md(:, :, 1, sampleIndex) = single(sample.mdMap);
    labels(sampleIndex) = uint8(find(cfg.ClassNames == manifest.label(sampleIndex), 1));
    clutterDB(sampleIndex) = sample.metadata.Scenario.ClutterToTargetDB;
end

data.RD = rd;
data.MD = md;
data.Labels = labels;
data.SNRDB = manifest.snr_db;
data.ClutterDB = clutterDB;
data.SampleID = manifest.sample_id;
data.Split = manifest.split;
data.Manifest = manifest;
data.TrainIndices = find(manifest.split == "train");
data.ValidationIndices = find(manifest.split == "validation");
data.TestIndices = find(manifest.split == "test");

fprintf("Loaded train/validation/test counts: %d/%d/%d.\n", ...
    numel(data.TrainIndices), numel(data.ValidationIndices), numel(data.TestIndices));
end
