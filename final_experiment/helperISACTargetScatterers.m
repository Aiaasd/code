function [positions, coefficients, truth] = ...
    helperISACTargetScatterers(label, time, scenario, cfg)
%helperISACTargetScatterers Unified bird, drone, and vehicle point model.
%   POSITIONS is 3-by-K-by-N and COEFFICIENTS is K-by-N. Bird and drone
%   kinematics are adapted from MathWorks-Radar-Drone-Project. Vehicle
%   contour, azimuth RCS, and rotating wheels are adapted from Car.m in
%   FMCW_Radar_Target_Simulator.

arguments
    label (1,1) string
    time (1,:) double {mustBeFinite, mustBeNonnegative}
    scenario (1,1) struct
    cfg (1,1) struct
end

switch label
    case "bird"
        [positions, coefficients] = birdScatterers(time, scenario, cfg);
    case "drone"
        [positions, coefficients] = droneScatterers(time, scenario, cfg);
    case "vehicle"
        [positions, coefficients] = vehicleScatterers(time, scenario, cfg);
    otherwise
        error("Unknown target class: %s", label);
end

centerTrack = targetCenter(time, scenario);
dTx = vecnorm(centerTrack-cfg.TransmitPosition, 2, 1);
dRx = vecnorm(centerTrack-cfg.ReceivePosition, 2, 1);
truth.Label = label;
truth.CenterTrack = centerTrack;
truth.CenterVelocity = scenario.CenterVelocity;
truth.EquivalentRange = (dTx+dRx)/2;
truth.ComponentCount = size(positions, 2);
end

function [positions, coefficients] = birdScatterers(time, scenario, cfg)
rotation = motionFrame(scenario.CenterVelocity);
local = scenario.BirdScale*birdGeometry(time, scenario.MicroMotionPhase, scenario.BirdFlapHz);
center = targetCenter(time, scenario);
numTime = numel(time);
positions = zeros(3, 5, numTime);
for k = 1:5
    positions(:, k, :) = reshape(center + rotation*squeeze(local(:, k, :)), 3, 1, numTime);
end

semiAxes = [0.10 0.05 0.05 0.05 0.05; ...
            0.10 0.05 0.05 0.05 0.05; ...
            1.00 0.25 0.25 0.25 0.25];
look = bistaticLook(scenario.CenterPosition, cfg);
axesLocal = [1 0 0 0 0; 0 1 1 -1 -1; 0 0 0 0 0];
axesGlobal = rotation*axesLocal;
coefficients = zeros(5, numTime);
for k = 1:5
    axisK = axesGlobal(:, k)/max(norm(axesGlobal(:, k)), eps);
    theta = acos(min(max(abs(dot(axisK, look)), 0), 1));
    a = semiAxes(1, k); b = semiAxes(2, k); c = semiAxes(3, k);
    rcs = pi*a^2*b^2*c^2/(a^2*sin(theta)^2+c^2*cos(theta)^2)^2;
    coefficients(k, :) = sqrt(rcs)*exp(1i*0.37*(k-1));
end
end

function local = birdGeometry(time, phase0, f)
ksi1 = 40*cos(2*pi*f*time+phase0) + 15;
ksi2 = 30*cos(2*pi*f*time+phase0) + 40;
theta2 = 20*sin(2*pi*f*time+phase0);
L1 = 0.5; L2 = 0.5;

elbow = [zeros(size(time)); L1*cosd(ksi1); L1*sind(ksi1)];
y2 = L1*cosd(ksi1) + L2*cosd(theta2).*cosd(ksi1-ksi2);
twist = theta2./cosd(ksi1-ksi2);
x2 = -(y2-elbow(2, :)).*tand(twist);
z2 = elbow(3, :) + (y2-elbow(2, :)).*tand(ksi1-ksi2);
wrist = [x2; y2; z2];
rightElbow = [elbow(1, :); -elbow(2, :); elbow(3, :)];
rightWrist = [wrist(1, :); -wrist(2, :); wrist(3, :)];

local = zeros(3, 5, numel(time));
local(:, 1, :) = reshape(zeros(3, numel(time)), 3, 1, []);
local(:, 2, :) = reshape(elbow/2, 3, 1, []);
local(:, 3, :) = reshape((elbow+wrist)/2, 3, 1, []);
local(:, 4, :) = reshape(rightElbow/2, 3, 1, []);
local(:, 5, :) = reshape((rightElbow+rightWrist)/2, 3, 1, []);
end

function [positions, coefficients] = droneScatterers(time, scenario, cfg)
rotation = motionFrame(scenario.CenterVelocity);
center = targetCenter(time, scenario);
rotorCenters = scenario.DroneScale*[0.25 0 -0.25 0; 0 0.25 0 -0.25; 0 0 0 0];
rotorRates = scenario.RotorRateScale*[40 -42 38 -41];
bladeRadius = scenario.DroneScale*(0.05+0.18)/2;
bladeLength = scenario.DroneScale*(0.18-0.05);
numTime = numel(time);
positions = zeros(3, 9, numTime);
positions(:, 1, :) = reshape(center, 3, 1, []);
coefficients = zeros(9, numTime);
coefficients(1, :) = 0.20;
look = bistaticLook(scenario.CenterPosition, cfg);
beta = asin(min(max(abs(dot([0; 0; 1], look)), 0), 1));
lambda = physconst("LightSpeed")/cfg.CarrierFrequency;

component = 2;
for rotor = 1:4
    for blade = 1:2
        phase = 2*pi*rotorRates(rotor)*time + ...
            scenario.MicroMotionPhase + pi*(blade-1);
        local = rotorCenters(:, rotor) + bladeRadius*[cos(phase); sin(phase); zeros(size(phase))];
        positions(:, component, :) = reshape(center + rotation*local, 3, 1, []);
        x = (4*pi/lambda)*(bladeLength/2)*cos(beta).*sin(phase);
        coefficients(component, :) = bladeLength*sinc(x/pi).*exp(1i*0.53*(component-2));
        component = component + 1;
    end
end
end

function [positions, coefficients] = vehicleScatterers(time, scenario, cfg)
if scenario.VehicleType == 0
    width = 1.8; lengthVehicle = 4.5; height = 1.5; tireRadius = 0.30;
    rcsTable = [10 7 6 5 4 4 3 2 5 20 5 0 0 0 1 3 5 10 15];
else
    width = 2.01; lengthVehicle = 5.8; height = 2.1; tireRadius = 0.375;
    rcsTable = 4 + [10 7 6 5 4 4 3 2 5 20 5 0 0 0 1 3 5 10 15];
end

rotation = motionFrame(scenario.CenterVelocity);
width=width*scenario.VehicleScale; lengthVehicle=lengthVehicle*scenario.VehicleScale;
height=height*scenario.VehicleScale; tireRadius=tireRadius*scenario.VehicleScale;
center = targetCenter(time, scenario);
numTime = numel(time);

% Sixteen evenly spaced body contour points reproduce the rectangular
% scattering outline in Car.m without importing its FMCW dependencies.
s = linspace(0, 1, 5); s(end) = [];
front = [lengthVehicle/2*ones(1,4); -width/2+width*s; height*0.65*ones(1,4)];
left = [lengthVehicle/2-lengthVehicle*s; width/2*ones(1,4); height*0.55*ones(1,4)];
back = [-lengthVehicle/2*ones(1,4); width/2-width*s; height*0.55*ones(1,4)];
right = [-lengthVehicle/2+lengthVehicle*s; -width/2*ones(1,4); height*0.55*ones(1,4)];
bodyLocal = [front left back right];

wheelCenters = [ lengthVehicle/3 lengthVehicle/3 -lengthVehicle/3 -lengthVehicle/3; ...
                 width/2 -width/2 width/2 -width/2; ...
                 tireRadius tireRadius tireRadius tireRadius];
numBody = size(bodyLocal, 2);
numWheel = 8;
positions = zeros(3, numBody+numWheel, numTime);
for k = 1:numBody
    positions(:, k, :) = reshape(center + rotation*bodyLocal(:, k), 3, 1, []);
end

omega = norm(scenario.CenterVelocity)/tireRadius;
angularAccel = dot(scenario.CenterAcceleration,scenario.CenterVelocity/max(norm(scenario.CenterVelocity),eps))/tireRadius;
component = numBody+1;
for wheel = 1:4
    for rim = 1:2
        phase = omega*time + 0.5*angularAccel*time.^2 + scenario.MicroMotionPhase + pi*(rim-1);
        local = wheelCenters(:, wheel) + [tireRadius*sin(phase); zeros(size(phase)); tireRadius*cos(phase)];
        positions(:, component, :) = reshape(center + rotation*local, 3, 1, []);
        component = component+1;
    end
end

look = bistaticLook(scenario.CenterPosition, cfg);
headingVector = scenario.CenterVelocity/max(norm(scenario.CenterVelocity), eps);
aspect = acosd(min(max(abs(dot(headingVector, look)), 0), 1));
rcsDBsm = interp1(0:10:180, rcsTable, aspect, "linear", "extrap");
totalRCS = 10^(rcsDBsm/10);
weights = [ones(1, numBody) 4*ones(1, numWheel)];
weights = weights/sum(weights);
coefficients = sqrt(totalRCS*weights(:)).*exp(1i*0.19*(0:numBody+numWheel-1).');
coefficients = repmat(coefficients, 1, numTime);
end

function center = targetCenter(time, scenario)
center = scenario.CenterPosition + scenario.CenterVelocity*time + 0.5*scenario.CenterAcceleration*time.^2;
end

function rotation = motionFrame(velocity)
forward = velocity/max(norm(velocity), eps);
upReference = [0; 0; 1];
if abs(dot(forward, upReference)) > 0.99
    upReference = [0; 1; 0];
end
lateral = cross(upReference, forward);
lateral = lateral/max(norm(lateral), eps);
up = cross(forward, lateral);
rotation = [forward lateral up];
end

function direction = bistaticLook(position, cfg)
toTx = cfg.TransmitPosition-position;
toRx = cfg.ReceivePosition-position;
toTx = toTx/max(norm(toTx), eps);
toRx = toRx/max(norm(toRx), eps);
direction = toTx+toRx;
direction = direction/max(norm(direction), eps);
end
