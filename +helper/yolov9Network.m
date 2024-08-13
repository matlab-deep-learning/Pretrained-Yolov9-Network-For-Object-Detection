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
% Load Yolov9c Detector
detector = helper.downloadPretrainedYOLOv9('Yolov9c');
end

if strcmpi(modelname,'Yolov9e')
% Load Yolov9e Detector
detector = helper.downloadPretrainedYOLOv9('Yolov9e');
end

if strcmpi(modelname,'Yolov9t')
% Load Yolov9t Detector
detector = helper.downloadPretrainedYOLOv9('Yolov9t');
end

if strcmpi(modelname,'Yolov9s')
% Load Yolov9s Detector
detector = helper.downloadPretrainedYOLOv9('Yolov9s');
end

if strcmpi(modelname,'Yolov9m')
% Load Yolov9m Detector
detector = helper.downloadPretrainedYOLOv9('Yolov9m');
end

detector = detector.net;