function bbox = xyxy2xywh(bbox)
% xyxy2xywh converts bounding boxes from [x1, y1, x2, y2] format to [x, y, width, height] format.
% Here, (x1, y1) is the top-left corner, and (x2, y2) is the bottom-right corner.
%
% Inputs:
% bbox         - An M-by-4 matrix of bounding boxes in [x1, y1, x2, y2] format.
%
% Outputs:
% bbox         - The converted bounding boxes in [x, y, width, height] format.

% Copyright 2024 The MathWorks, Inc

bbox(:,3) = bbox(:,3) - bbox(:,1) + 1; %1-based indexing
bbox(:,4) = bbox(:,4) - bbox(:,2) + 1; %1-based indexing
end