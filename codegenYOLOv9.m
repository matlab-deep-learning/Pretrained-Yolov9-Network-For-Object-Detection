% Read test image.
I = imread('inputTeam.jpg');

% Replace 'Yolov9m' with other supported versions here and in yolov9Predict to generate code for
% other YOLO v9 variants.
modelName = 'Yolov9m';
% Check if the model file exists
files = dir('**/*');
filename = strcat(modelName,'.mat');
fileExists = any(strcmp({files.name}, filename));
% Download model if it does not exists
if ~fileExists
    helper.downloadPretrainedYOLOv9(modelName);
end

% Display yolov9Predict function.
type('yolov9Predict.m');

% Generate MATLAB code.
cfg = coder.config('mex');
cfg.TargetLang = 'C++';
% % 'cudnn' and 'none' are also supported.
cfg.DeepLearningConfig = coder.DeepLearningConfig(TargetLibrary = 'mkldnn');
inputArgs = {I};
codegen -config cfg yolov9Predict -args inputArgs -report

% Perform detection using pretrained model.
[bboxes,scores,labelIds] = yolov9Predict_mex(I);

% Plot Detections
helper.plotObjectDetections(I,bboxes,scores,labelIds);