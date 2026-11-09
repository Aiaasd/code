function manifest = generateISACDataset(cfg)
%generateISACDataset Generate paired RD/MD samples for three target classes.

arguments
    cfg (1,1) struct = isacDatasetConfig()
end

outputRoot = cfg.OutputRoot;
if ~isfolder(outputRoot)
    mkdir(outputRoot);
end
for split = ["train" "validation" "test"]
    for label = cfg.Classes
        folder = fullfile(outputRoot, split, label);
        if ~isfolder(folder)
            mkdir(folder);
        end
    end
end
for feature = ["rd" "md"]
    for label = cfg.Classes
        folder = fullfile(outputRoot, "images", feature, label);
        if ~isfolder(folder)
            mkdir(folder);
        end
    end
end

totalSamples = numel(cfg.Classes)*cfg.SamplesPerClass;
sampleID = strings(totalSamples, 1);
labelColumn = strings(totalSamples, 1);
splitColumn = strings(totalSamples, 1);
snrColumn = zeros(totalSamples, 1);
matPath = strings(totalSamples, 1);
rdImagePath = strings(totalSamples, 1);
mdImagePath = strings(totalSamples, 1);
detectedColumn = false(totalSamples, 1);
rangeColumn = zeros(totalSamples, 1);
dopplerColumn = zeros(totalSamples, 1);
seedColumn = zeros(totalSamples, 1);

row = 0;
startTime = tic;
for classIndex = 1:numel(cfg.Classes)
    label = cfg.Classes(classIndex);
    for snrIndex = 1:numel(cfg.SNRLevelsDB)
        snrDB = cfg.SNRLevelsDB(snrIndex);
        for localIndex = 1:cfg.SamplesPerSNR
            classSampleIndex = (snrIndex-1)*cfg.SamplesPerSNR+localIndex;
            row = row+1;
            split = splitForIndex(localIndex, cfg.SamplesPerSNR, cfg);
            id = sprintf("%s_%04d", label, classSampleIndex);

            sensing = simulateSensingSample(label, snrDB, classSampleIndex, cfg);
            products = processSensingSample(sensing, cfg);

            payload.frequencyResponse = sensing.FrequencyResponse;
            payload.rdMap = products.RangeDopplerMap;
            payload.mdMap = products.MicroDopplerMap;
            metadata.Label = label;
            metadata.Split = split;
            metadata.SampleID = string(id);
            metadata.SNRDB = snrDB;
            metadata.NoiseVariance = sensing.NoiseVariance;
            metadata.Scenario = sensing.Scenario;
            metadata.Truth = sensing.Truth;
            metadata.Detection = products.Detection;
            metadata.System.CarrierFrequency = cfg.CarrierFrequency;
            metadata.System.Bandwidth = cfg.Bandwidth;
            metadata.System.OfficialNumSubcarriers = cfg.OfficialNumSubcarriers;
            metadata.System.StoredFrequencyBins = cfg.NumFrequencyBins;
            metadata.System.SlowTimeRate = cfg.SlowTimeRate;
            metadata.System.ObservationDuration = cfg.ObservationDuration;

            relativeMat = fullfile(split, label, id+".mat");
            relativeRD = fullfile("images", "rd", label, id+".png");
            relativeMD = fullfile("images", "md", label, id+".png");
            payload.metadata = metadata;
            save(fullfile(outputRoot, relativeMat), "-struct", "payload", "-v7");
            imwrite(uint8(255*payload.rdMap), fullfile(outputRoot, relativeRD));
            imwrite(uint8(255*payload.mdMap), fullfile(outputRoot, relativeMD));

            sampleID(row) = id;
            labelColumn(row) = label;
            splitColumn(row) = split;
            snrColumn(row) = snrDB;
            matPath(row) = string(relativeMat);
            rdImagePath(row) = string(relativeRD);
            mdImagePath(row) = string(relativeMD);
            detectedColumn(row) = products.Detection.Detected;
            rangeColumn(row) = products.Detection.RangeMeters;
            dopplerColumn(row) = products.Detection.DopplerHz;
            seedColumn(row) = sensing.Scenario.Seed;
            clear payload metadata

            if mod(row, 5) == 0 || row == totalSamples
                fprintf("Generated %d/%d samples (%.1f min).\n", ...
                    row, totalSamples, toc(startTime)/60);
            end
        end
    end
end

manifest = table(sampleID, labelColumn, splitColumn, snrColumn, ...
    detectedColumn, rangeColumn, dopplerColumn, seedColumn, ...
    matPath, rdImagePath, mdImagePath, ...
    VariableNames=["sample_id" "label" "split" "snr_db" "detected" ...
    "detected_range_m" "detected_doppler_hz" "seed" "mat_path" ...
    "rd_image_path" "md_image_path"]);
writetable(manifest, fullfile(outputRoot, "manifest.csv"));
save(fullfile(outputRoot, "dataset_config.mat"), "cfg");
end

function split = splitForIndex(index, count, cfg)
trainEnd = round(cfg.TrainFraction*count);
validationEnd = trainEnd+round(cfg.ValidationFraction*count);
if index <= trainEnd
    split = "train";
elseif index <= validationEnd
    split = "validation";
else
    split = "test";
end
end
