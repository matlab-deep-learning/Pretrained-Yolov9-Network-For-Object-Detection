function plotObjectDetections(img,bboxes,scores,labelIds)
% plotObjectDetections plots object detection results on an input image.
%
% Inputs:
% img         - Input RGB image.
% bboxes      - Bounding boxes of detected objects, each row is [x, y, width, height].
% scores      - Confidence scores for each detected object.
% labelIds    - Class IDs for each detected object.

% Copyright 2024 The MathWorks, Inc.

% Obtain class names from COCO dataset
classNames = helper.getCOCOClassNames();

% Map labelIds back to labels.
labels = classNames(labelIds);

% Set font size for annotations
fontSize = 14; 

% Specify colors for text and boxes for better visibility and aesthetics
textColor = 'white'; % Text color
textBackgroundColor = 'blue'; % Background color for text for better visibility

% Construct annotations by combining labels and scores
annotations = string(labels) + ': ' + string(scores);

% Insert annotations with specified colors and increased font size
Iout = insertObjectAnnotation(img,'rectangle',bboxes,annotations,'FontSize',fontSize,'LineWidth', 3, ...
    'TextColor', textColor, 'TextBoxOpacity', 0.6, 'AnnotationColor', textBackgroundColor);

% Display the annotated image
figure, imshow(Iout);
end


