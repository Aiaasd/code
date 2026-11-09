function scenario = createRandomTargetScenario(label, sampleIndex, cfg)
%createRandomTargetScenario Draw one reproducible target and clutter scene.

arguments
    label (1,1) string
    sampleIndex (1,1) double {mustBeInteger, mustBePositive}
    cfg (1,1) struct
end

classIndex = find(cfg.Classes == label, 1);
if isempty(classIndex)
    error("Unknown target class: %s", label);
end

seed = cfg.RandomSeed + 10000*classIndex + sampleIndex;
rng(seed, "twister");

% Keep targets away from both stations while varying bistatic geometry.
for attempt = 1:100
    center = [35 + 75*rand; -45 + 90*rand; 0];
    if norm(center-cfg.TransmitPosition) > 25 && ...
            norm(center-cfg.ReceivePosition) > 25
        break
    end
end

switch label
    case "bird"
        center(3) = 8 + 18*rand;
        speed = 3 + 13*rand;
    case "drone"
        center(3) = 6 + 20*rand;
        speed = 1 + 15*rand;
    case "vehicle"
        center(3) = 0.8;
        speed = 1 + 15*rand;
end

heading = -pi + 2*pi*rand;
velocity = speed*[cos(heading); sin(heading); 0];

scenario.Label = label;
scenario.SampleIndex = sampleIndex;
scenario.Seed = seed;
scenario.CenterPosition = center;
scenario.CenterVelocity = velocity;
scenario.CenterAcceleration = (2*rand-1)*1.5*[cos(heading);sin(heading);0];
scenario.BirdFlapHz = 3 + 5*rand;
scenario.BirdScale = 0.7 + 0.6*rand;
scenario.RotorRateScale = 0.75 + 0.5*rand;
scenario.DroneScale = 0.8 + 0.4*rand;
scenario.VehicleScale = 0.85 + 0.3*rand;
if isfield(cfg,'ChallengeMode') && cfg.ChallengeMode
    scenario.BirdFlapHz = 2 + 7*rand;
    scenario.BirdScale = 0.6 + 0.8*rand;
    scenario.RotorRateScale = 0.65 + 0.7*rand;
    scenario.DroneScale = 0.7 + 0.6*rand;
    scenario.VehicleScale = 0.75 + 0.5*rand;
end
scenario.Heading = heading;
scenario.MicroMotionPhase = 2*pi*rand;
scenario.VehicleType = double(rand >= 0.5);
scenario.ClutterToTargetDB = cfg.ClutterToTargetRangeDB(1) + ...
    diff(cfg.ClutterToTargetRangeDB)*rand;
end
