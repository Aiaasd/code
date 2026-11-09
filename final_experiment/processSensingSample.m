function products = processSensingSample(sample, cfg)
%processSensingSample Form RD/MD maps and run two-dimensional CA-CFAR.

c = physconst("LightSpeed");
response = double(sample.FrequencyResponse);
response = response-mean(response, 2);
rangeTime = ifft(ifftshift(response, 1), cfg.NumFrequencyBins, 1);

slowWindow = hann(cfg.NumSlowTimeSamples, "periodic").';
rdComplex = fftshift(fft(rangeTime.*slowWindow, cfg.NumSlowTimeSamples, 2), 2);
rdPower = abs(rdComplex).^2;
rangeAxis = (0:cfg.NumFrequencyBins-1)*c/(2*cfg.Bandwidth);
dopplerAxis = (-cfg.NumSlowTimeSamples/2:cfg.NumSlowTimeSamples/2-1)* ...
    cfg.SlowTimeRate/cfg.NumSlowTimeSamples;

[detected, rangeIndex, dopplerIndex, threshold] = cfarPeak(rdPower, rangeAxis, cfg);
gateSource = "cfar";
if ~detected
    [~, rangeIndex] = min(abs(rangeAxis-median(sample.Truth.EquivalentRange)));
    [~, dopplerIndex] = max(rdPower(rangeIndex, :));
    gateSource = "truth_fallback";
end

% Align the moving target's range ridge before STFT. During one second the
% target can migrate through several 1.5 m cells; a fixed gate would keep
% only part of the micro-motion record. Simulation truth supplies the
% coarse ridge here, playing the role of a range tracker in measured data.
rangeResolution = c/(2*cfg.Bandwidth);
trackIndices = round(double(sample.Truth.EquivalentRange)/rangeResolution)+1;
trackIndices = min(max(trackIndices, 1), cfg.NumFrequencyBins);
slowSignal = complex(zeros(1, cfg.NumSlowTimeSamples));
for timeIndex = 1:cfg.NumSlowTimeSamples
    gate = max(1, trackIndices(timeIndex)-2): ...
        min(cfg.NumFrequencyBins, trackIndices(timeIndex)+2);
    slowSignal(timeIndex) = sum(rangeTime(gate, timeIndex));
end
slowSignal = slowSignal-mean(slowSignal);
bulkDoppler = dopplerAxis(dopplerIndex);
slowSignal = slowSignal.*exp(-1i*2*pi*bulkDoppler*double(sample.Time));

win = hann(cfg.STFTWindowLength, "periodic");
[stftCoefficients, mdFrequency, mdTime] = stft(slowSignal, cfg.SlowTimeRate, ...
    Window=win, OverlapLength=cfg.STFTOverlapLength, ...
    FFTLength=cfg.STFTFFTLength, FrequencyRange="centered");
mdPower = abs(stftCoefficients).^2/sum(win)^2;
% Suppress the residual bulk-motion ridge so that wing, rotor, and wheel
% sidebands determine the micro-Doppler image contrast.
notch = abs(mdFrequency) <= 2*cfg.SlowTimeRate/cfg.STFTWindowLength;
referenceRows = abs(mdFrequency) > 4*cfg.SlowTimeRate/cfg.STFTWindowLength;
replacement = median(mdPower(referenceRows, :), 1);
mdPower(notch, :) = repmat(replacement, sum(notch), 1);

rdNormalized = normalizeLogPower(rdPower, cfg.DynamicRangeDB);
mdNormalized = normalizeLogPower(mdPower, cfg.DynamicRangeDB);
rdMap = resizeFeatureMap(rdNormalized, cfg.RangeDopplerOutputSize);
mdMap = resizeFeatureMap(mdNormalized, cfg.MicroDopplerOutputSize);

products.RangeDopplerMap = single(rdMap);
products.MicroDopplerMap = single(mdMap);
products.RangeAxis = single(rangeAxis);
products.DopplerAxis = single(dopplerAxis);
products.MicroDopplerFrequencyAxis = single(mdFrequency);
products.MicroDopplerTimeAxis = single(mdTime);
products.Detection.Detected = detected;
products.Detection.RangeMeters = rangeAxis(rangeIndex);
products.Detection.DopplerHz = dopplerAxis(dopplerIndex);
products.Detection.RangeIndex = rangeIndex;
products.Detection.DopplerIndex = dopplerIndex;
products.Detection.Threshold = threshold;
products.Detection.GateSource = gateSource;
end

function [detected, rangeIndex, dopplerIndex, selectedThreshold] = ...
    cfarPeak(powerMap, rangeAxis, cfg)
trainingSize = cfg.CFARTrainingSize;
guardSize = cfg.CFARGuardSize;
kernel = ones(trainingSize);
rowCenter = (trainingSize(1)+1)/2;
colCenter = (trainingSize(2)+1)/2;
rowGuard = (guardSize(1)-1)/2;
colGuard = (guardSize(2)-1)/2;
kernel(rowCenter-rowGuard:rowCenter+rowGuard, ...
       colCenter-colGuard:colCenter+colGuard) = 0;
numTraining = sum(kernel(:));
noiseEstimate = conv2(powerMap, kernel/numTraining, "same");
alpha = numTraining*(cfg.CFARProbabilityFalseAlarm^(-1/numTraining)-1);
thresholdMap = alpha*noiseEstimate;

validRange = rangeAxis >= 5 & rangeAxis <= 180;
detections = powerMap > thresholdMap;
detections(~validRange, :) = false;
score = powerMap./max(thresholdMap, eps);
score(~detections) = 0;
[bestScore, linearIndex] = max(score(:));
detected = bestScore > 1;
if detected
    [rangeIndex, dopplerIndex] = ind2sub(size(powerMap), linearIndex);
else
    powerMap(~validRange, :) = 0;
    [~, linearIndex] = max(powerMap(:));
    [rangeIndex, dopplerIndex] = ind2sub(size(powerMap), linearIndex);
end
selectedThreshold = thresholdMap(rangeIndex, dopplerIndex);
end

function normalized = normalizeLogPower(powerMap, dynamicRangeDB)
powerDB = 10*log10(powerMap/max(powerMap(:))+eps);
powerDB = max(powerDB, -dynamicRangeDB);
normalized = (powerDB+dynamicRangeDB)/dynamicRangeDB;
end

function output = resizeFeatureMap(input, outputSize)
[oldRows, oldCols] = size(input);
[xOld, yOld] = meshgrid(1:oldCols, 1:oldRows);
[xNew, yNew] = meshgrid(linspace(1, oldCols, outputSize(2)), ...
    linspace(1, oldRows, outputSize(1)));
output = interp2(xOld, yOld, input, xNew, yNew, "linear");
output = min(max(output, 0), 1);
end
