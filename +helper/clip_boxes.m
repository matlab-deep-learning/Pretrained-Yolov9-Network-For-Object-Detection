function boxes = clip_boxes(boxes, img_shape)
% clip_boxes ensures bounding boxes are within the image boundaries.
%
% Inputs:
% boxes           -   A Nx4 matrix of bounding boxes in the format [x1, y1, x2, y2],
%                     where (x1, y1) is the top-left corner, and (x2, y2) is the
%                     bottom-right corner of the bounding box.
% img_shape       -   A vector specifying the image size
%
% Outputs:
% boxes           -   The input matrix of bounding boxes, adjusted so that all
%                     bounding boxes fit within the specified image boundaries.

% Copyright 2024 The MathWorks, Inc

% Ensure x-coordinates are within the width boundary of the image
boxes(:,1) = max(min(boxes(:,1), img_shape(2)), 0.1);
boxes(:,3) = max(min(boxes(:,3), img_shape(2)), 0.1);

% Ensure y-coordinates are within the height boundary of the image
boxes(:,2) = max(min(boxes(:,2), img_shape(1)), 0.1);
boxes(:,4) = max(min(boxes(:,4), img_shape(1)), 0.1);
end