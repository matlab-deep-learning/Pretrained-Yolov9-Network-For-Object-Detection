function final_output = postprocess(featureMap, preprocessed_image, original_image)
% postprocess processes the feature map from YOLOv9 detection and returns final bounding boxes, confidence scores, and class IDs.
%
% Inputs:
% featureMap            - The output feature map from the YOLOv9 detector.
% preprocessed_image    - The image after being pre-processed for YOLOv9 input.
% original_image        - The original image before any processing.
%
% Outputs:
% final_output          - An array containing the final bounding boxes, confidence scores, and class IDs. Each row corresponds to a detected object.

% Copyright 2024 The MathWorks, Inc

% Initialization and Configuration
conf_thresh = 0.25; %Confidence threshold
%conf_thresh = 0.01; %Try this for Evaluation
iou_thres = 0.70; % IoU threshold for NMS
num_classes = 80; % Total number of classes

% Pre-process Predictions
pred = permute(featureMap, [2,1]);

% Extract class scores and find max values
class_scores = max(pred(:, 5:5+num_classes-1), [], 2);
conf_mask = class_scores > conf_thresh; % Mask for detections above the confidence threshold
pred = pred(conf_mask, :); %Fiter predictions based on Confidence Threshold

box = pred(:,1:4);
cls = pred(:,5:5+num_classes-1);

% Convert bounding boxes from [x_center, y_center, width, height] to [x1, y1, w, h] for NMS
box = helper.XcYcwh2xywh(box);

% Filter out class with maximum confidence probability
[classProbs, classIdx] = max(cls,[],2);
scorePred = classProbs;
classPred = classIdx;

% Filtered detections based on confidence
filtered_boxes = box;
filtered_scores = scorePred;
filtered_classes = classPred; 

filtered_scores = double(extractdata(filtered_scores));
filtered_classes = double(extractdata(filtered_classes));


% Apply NMS
[selectedBboxes, selectedScores, selectedClasses] = selectStrongestBboxMulticlass(filtered_boxes, filtered_scores,filtered_classes, ...
    'RatioType', 'Union', 'OverlapThreshold', iou_thres);

% Prepare Final Output
% Combine NMS survived boxes, scores, and classes
final_output = [selectedBboxes, selectedScores, selectedClasses];

% Post-process bounding boxes to map back to original image dimensions
final_output(:,1:4) = helper.xywh2xyxy(final_output(:,1:4)); % Converting [x,y,w,h] to [x1,y1,x2,y2] for scaling
final_output(:,1:4) = helper.scale_boxes(size(preprocessed_image), final_output(:,1:4), size(original_image)); % Map the final bboxes to original dimensions

% Obtain the final bounding boxes in standard [x,y,w,h] format
final_output(:,1:4) = helper.xyxy2xywh(final_output(:,1:4));
final_output = gather(final_output);

end