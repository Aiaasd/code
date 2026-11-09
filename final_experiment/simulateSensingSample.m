function sample = simulateSensingSample(label, snrDB, sampleIndex, cfg)
%simulateSensingSample Generate one long coherent MIMO-OFDM sensing sample.
%   The saved frequency response is a uniformly decimated view of the
%   official 2048-subcarrier, 100 MHz channel grid. It is the sufficient
%   sensing statistic after removal of known transmitted OFDM symbols.

arguments
    label (1,1) string
    snrDB (1,1) double {mustBeFinite}
    sampleIndex (1,1) double {mustBeInteger, mustBePositive}
    cfg (1,1) struct
end

scenario = createRandomTargetScenario(label, sampleIndex, cfg);
rng(scenario.Seed, "twister");
c = physconst("LightSpeed");
time = (0:cfg.NumSlowTimeSamples-1)/cfg.SlowTimeRate;
frequencyOffset = (-cfg.NumFrequencyBins/2:cfg.NumFrequencyBins/2-1).' * ...
    (cfg.Bandwidth/cfg.NumFrequencyBins);

[positions, coefficients, truth] = ...
    helperISACTargetScatterers(label, time, scenario, cfg);
numComponents = size(positions, 2);
targetResponse = complex(zeros(cfg.NumFrequencyBins, cfg.NumSlowTimeSamples));

for k = 1:numComponents
    pointTrack = squeeze(positions(:, k, :));
    dTx = vecnorm(pointTrack-cfg.TransmitPosition, 2, 1);
    dRx = vecnorm(pointTrack-cfg.ReceivePosition, 2, 1);
    delay = (dTx+dRx)/c;
    rangeLoss = 1./sqrt(max(dTx.*dRx, 1));
    carrierTerm = coefficients(k, :).*rangeLoss.* ...
        exp(-1i*2*pi*cfg.CarrierFrequency*delay);
    targetResponse = targetResponse + ...
        exp(-1i*2*pi*(frequencyOffset*delay)).*carrierTerm;
end

targetPower = mean(abs(targetResponse(:)).^2);
targetResponse = targetResponse/sqrt(max(targetPower, eps));

clutterResponse = complex(zeros(size(targetResponse)));
numClutter = randi(cfg.ClutterCountRange);
for k = 1:numClutter
    equivalentRange = 15 + 150*rand;
    delay = 2*equivalentRange/c;
    clutterDoppler = -3 + 6*rand;
    amplitude = (0.25+0.75*rand)*exp(1i*2*pi*rand);
    clutterResponse = clutterResponse + amplitude * ...
        exp(-1i*2*pi*frequencyOffset*delay) * ...
        exp(1i*2*pi*clutterDoppler*time);
end
clutterPower = mean(abs(clutterResponse(:)).^2);
clutterResponse = clutterResponse/sqrt(max(clutterPower, eps));
clutterResponse = clutterResponse*10^(scenario.ClutterToTargetDB/20);

cleanResponse = targetResponse + clutterResponse;
% Target response has unit average power, so 0 dBW is the explicit signal
% reference used to set the requested target-to-noise ratio.
[frequencyResponse, noiseVariance] = awgn(cleanResponse, snrDB, 0);

sample.FrequencyResponse = complex(single(frequencyResponse));
sample.Time = single(time);
sample.FrequencyOffset = single(frequencyOffset);
sample.NoiseVariance = noiseVariance;
sample.SNRDB = snrDB;
sample.Scenario = scenario;
sample.Scenario.NumClutterScatterers = numClutter;
sample.Truth = truth;
end
