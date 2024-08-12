function bbox = xywh2xyxy(bbox)
% xywh2xyxy converts bounding boxes from [x, y, width, height] format to [x1, y1, x2, y2] format.
% Here, (x, y) is the top-left corner, and (x2, y2) is the bottom-right corner.
%
% Inputs:
% bbox      - An M-by-4 matrix of bounding boxes in [x, y, width, height] format.
%
% Outputs:
% bbox      - The converted bounding boxes in [x1, y1, x2, y2] format.

% Copyright 2024 The MathWorks, Inc

% width and height
dw = bbox(:, 3);
dh = bbox(:, 4);

% Calculate bottom right x and y
bbox(:, 3) = bbox(:, 1) + dw - 1; %1-based indexing
bbox(:, 4) = bbox(:, 2) + dh - 1; %1-based indexing
end