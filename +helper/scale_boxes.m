function boxes = scale_boxes(img1_shape, boxes, img0_shape,ratio_pad,padding,xywh)
% scale_boxes scales and adjusts bounding boxes from a processed image back to the original image size.
%
% Inputs:
% img1_shape            - Size of the processed image [height, width].
% boxes                 - Bounding boxes in the processed image in [x1, y1, x2, y2] format.
% img0_shape            - Size of the original image [height, width].
% ratio_pad (optional)  - Scaling factor and padding applied during preprocessing. 
%                         Format: [gain; pad_x, pad_y]. If not provided, it is calculated.
% padding (optional)    - Boolean indicating if padding adjustment is needed. Default is true.
% xywh (optional)       - Boolean indicating the format of the boxes. False for [x1, y1, x2, y2], 
%                         true for [x, y, width, height]. Default is false.
%
% Outputs:
% boxes                 - Scaled and adjusted bounding boxes in the format of the input 'boxes', 
%                         mapped back to the original image size.  

% Copyright 2024 The MathWorks, Inc

% Check for optional arguments
if nargin < 4 || isempty(ratio_pad)
    gain = min(img1_shape(1) / img0_shape(1), img1_shape(2) / img0_shape(2)); % gain = actual / original
    pad = [(img1_shape(2) - img0_shape(2)*gain) ./ 2,(img1_shape(1) - img0_shape(1)*gain) ./ 2]; % wh padding
else
    gain = ratio_pad(1,1);
    pad = ratio_pad(2);
end

if nargin < 5, padding = true; end
if nargin < 6, xywh = false; end

% Adjust boxes for padding
if padding
    boxes(:,1) = boxes(:,1) - pad(1);
    boxes(:,2) = boxes(:,2) - pad(2);
    if xywh==false
        boxes(:,3) = boxes(:,3) - pad(1); % x padding
        boxes(:,4) = boxes(:,4) - pad(2); % y padding
    end
end

% Scale boxes
boxes = boxes./ gain;
% Clip boxes to image boundaries
boxes = helper.clip_boxes(boxes, img0_shape);
end