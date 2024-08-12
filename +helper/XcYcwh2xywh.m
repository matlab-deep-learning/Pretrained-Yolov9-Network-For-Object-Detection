function bbox = XcYcwh2xywh(bbox)
% XcYcwh2xywh converts bounding boxes from center format [Xcenter, Ycenter, width, height] 
% to corner format [x1, y1, width, height], where (x1, y1) is the top-left corner.
%
% Inputs:
% bbox          - An Nx4 matrix of bounding boxes in [Xcenter, Ycenter, width, height] format.
%
% Outputs:
% bbox          - The converted bounding boxes in [x1, y1, width, height] format.

% Copyright 2024 The MathWorks, Inc

bbox = single(extractdata(bbox));

% Convert center coordinates (Xcenter, Ycenter) to top-left corner (x1, y1)
bbox(:,1) = bbox(:,1)- bbox(:,3)/2; 
bbox(:,2) = bbox(:,2)- bbox(:,4)/2;

% Round down the coordinates to the nearest integer to avoid fractional pixels
bbox = floor(bbox);

% Ensure bounding box coordinates are at least 1 to stay within image bounds
bbox(bbox<1)=1; 

end 