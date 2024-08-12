% Read test image.
I = imread('inputTeam.jpg');

% Replace 'yolov9m' with other supported versions in yolov9Predict to generate code for
% other YOLO v9 variants.
modelName = 'yolov9m';

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