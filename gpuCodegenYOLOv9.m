% Read test image.
I = imread('inputTeam.jpg');

% Replace 'yolov9m' with other supported models in yolov9Predict to generate code for
% other YOLO v9 variants.
modelName = 'yolov9m';

% Display yolov9Predict function.
type('yolov9Predict.m');

% Generate CUDA Mex.
cfg = coder.gpuConfig('mex');
cfg.TargetLang = 'C++';
cfg.DeepLearningConfig = coder.DeepLearningConfig('cudnn');
inputArgs = {I};
codegen -config cfg yolov9Predict -args inputArgs -report

% Perform detection using pretrained model.
[bboxes,scores,labelIds] = yolov9Predict_mex(I);

% Plot Detections
helper.plotObjectDetections(I,bboxes,scores,labelIds);