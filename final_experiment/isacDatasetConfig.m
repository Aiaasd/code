function cfg = isacDatasetConfig()
%isacDatasetConfig Configuration for the three-class sensing dataset.

base = isacDefaultConfig();

cfg.RandomSeed = 20261106;
cfg.Classes = ["bird" "drone" "vehicle"];
cfg.SNRLevelsDB = [-10 -5 0 5 10];
cfg.SamplesPerSNR = 120;
cfg.SamplesPerClass = numel(cfg.SNRLevelsDB)*cfg.SamplesPerSNR;

cfg.CarrierFrequency = base.CarrierFrequency;
cfg.Bandwidth = base.Bandwidth;
cfg.TransmitPosition = base.TransmitPosition;
cfg.ReceivePosition = base.ReceivePosition;

% The official 2048-subcarrier grid is uniformly decimated by eight for
% dataset generation. The retained bins still span the full 100 MHz, so
% the 1.5 m range resolution is preserved while generation stays tractable.
cfg.OfficialNumSubcarriers = base.NumSubcarriers;
cfg.NumFrequencyBins = 256;
cfg.FrequencyDecimation = cfg.OfficialNumSubcarriers/cfg.NumFrequencyBins;

cfg.SlowTimeRate = 1024;
cfg.NumSlowTimeSamples = 1024;
cfg.ObservationDuration = cfg.NumSlowTimeSamples/cfg.SlowTimeRate;
cfg.NumClutterScatterers = 8;
cfg.ClutterCountRange = [4 16];
cfg.ChallengeMode = false;
cfg.ClutterToTargetRangeDB = [-10 5];

cfg.RangeDopplerOutputSize = [128 128];
cfg.MicroDopplerOutputSize = [128 128];
cfg.STFTWindowLength = 256;
cfg.STFTOverlapLength = 192;
cfg.STFTFFTLength = 512;
cfg.CFARTrainingSize = [15 15];
cfg.CFARGuardSize = [5 5];
cfg.CFARProbabilityFalseAlarm = 1e-4;
cfg.DynamicRangeDB = 60;

cfg.TrainFraction = 0.70;
cfg.ValidationFraction = 0.15;
cfg.TestFraction = 0.15;
cfg.OutputRoot = fullfile(fileparts(mfilename("fullpath")), "dataset");
end
