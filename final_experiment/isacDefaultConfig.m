function cfg = isacDefaultConfig()
%isacDefaultConfig Return the baseline communication-centric ISAC settings.

cfg.RandomSeed = 0;

cfg.CarrierFrequency = 6e9;
cfg.Bandwidth = 100e6;
cfg.SampleRate = cfg.Bandwidth;
cfg.PeakPower = 1.0;
cfg.NoiseFigure = 3.0;
cfg.ReferenceTemperature = 290;

cfg.NumTransmitElements = 8;
cfg.NumReceiveElements = 8;
cfg.TransmitPosition = [0; 0; 0];
cfg.TransmitOrientationAxes = eye(3);
cfg.ReceivePosition = [80; 60; 0];
cfg.ReceiveOrientationAxes = rotz(-90);

cfg.MaximumPathLength = 300;
cfg.MaximumVelocity = 50;
cfg.RegionOfInterest = [0 120; -80 80];
cfg.NumStaticScatterers = 200;

cfg.NumSubcarriers = 2048;
cfg.NumGuardBandCarriers = [9; 8];
cfg.NumFrames = 24;
cfg.BitsPerSymbol = 6;
cfg.NumDataStreams = 2;
end

