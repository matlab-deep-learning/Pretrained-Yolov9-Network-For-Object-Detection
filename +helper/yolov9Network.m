function detector = yolov9Network(modelname)
% Yolov9Network loads a pre-trained network of Yolov9
% Inputs:
% modelname   - Version of Yolov9 to be loaded {Yolov9c,Yolov9e,Yolov9m,Yolov9s,Yolov9t}
% Outputs:
% detector    - dlnetwork of loaded model

% Copyright 2024 The MathWorks, Inc

if isempty(modelname) 
     error('Please pass a valid model name as arguement to Yolov9ObjectDetector!');
end

if strcmpi(modelname,'Yolov9c')
% Load Yolov9c Network
detector = load("Yolov9c.mat");
end

if strcmpi(modelname,'Yolov9e')
% Load Yolov9e Network
detector = load("Yolov9e.mat");
end

if strcmpi(modelname,'Yolov9t')
% Load Yolov9t Network
detector = load("Yolov9t.mat");
end

if strcmpi(modelname,'Yolov9s')
% Load Yolov9s Network
detector = load("Yolov9s.mat");
end

if strcmpi(modelname,'Yolov9m')
% Load Yolov9m Network
detector = load("Yolov9m.mat");
end

detector = detector.net;