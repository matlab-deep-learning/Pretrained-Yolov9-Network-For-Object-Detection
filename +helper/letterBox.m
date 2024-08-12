function img = letterBox(image, options)
% letterBox resizes an image and adds padding.
% This function is MATLAB equivalent of the Python LetterBox class, providing flexible image resizing
% and padding options for object detection, instance segmentation, and pose estimation models.
%
% Inputs:
% image     - The input image to be resized and padded, specified as a non-empty matrix.
% options   - A structure containing name-value pairs to specify additional properties:
%     - newShape: Desired [height, width] of the output image. Default is [640, 640].
%     - auto: Boolean flag to adjust padding automatically to maintain the aspect ratio. Default is false.
%     - scaleFill: Boolean flag to scale the image to fill the newShape completely, ignoring the aspect ratio. Default is false.
%     - scaleup: Boolean flag to allow scaling up of the image. Default is true.
%     - center: Boolean flag to center the image with padding. Default is true.
%     - stride: Stride size used in padding calculation. Default is 32.
%
% Outputs:
% img       - The resized and padded image.

% Copyright 2024 The MathWorks, Inc.

% Define input arguments with default values and validations
arguments
    image {mustBeNonempty}
    % Define name-value pairs directly
    options.newShape (1,2) {mustBeNumeric, mustBeNonempty} = [640, 640]
    options.auto (1,1) {logical} = false
    options.scaleFill (1,1) {logical} = false
    options.scaleup (1,1) {logical} = true
    options.center (1,1) {logical} = true
    options.stride (1,1) {mustBeNumeric, mustBeNonempty} = 32
    
end

% Extract options for readability
newShape = options.newShape;
auto = options.auto;
scaleFill = options.scaleFill;
scaleup = options.scaleup;
center = options.center;
stride = options.stride;

shape = size(image);
shape = shape(1:2); % Current shape [height, width]

% Scale ratio (new / old)
r = min(newShape(1) / shape(1), newShape(2) / shape(2));
if ~scaleup % Only scale down, do not scale up (for better val mAP)
    r = min(r, 1.0);
end

% Compute padding
ratio = [r, r]; % width, height ratios
newUnpad = [round(shape(2) * r), round(shape(1) * r)]; % new shape [width, height] without padding
dw = newShape(2) - newUnpad(1); % width padding
dh = newShape(1) - newUnpad(2); % height padding

if auto % minimum rectangle
    dw = mod(dw, stride);
    dh = mod(dh, stride);
elseif scaleFill % stretch
    dw = 0;
    dh = 0;
    newUnpad = newShape;
    ratio = [newShape(2) / shape(2), newShape(1) / shape(1)]; % width, height ratios
end

if center
    dw = dw / 2; % divide padding into 2 sides
    dh = dh / 2;
end

% Resize image
if shape~=newUnpad
    img = imresize(image, fliplr(newUnpad), 'bilinear',Antialiasing = false);
end

% Add padding
if center
    top = round(dh - 0.1);
    bottom = round(dh + 0.1);
    left = round(dw - 0.1);
    right = round(dw + 0.1);
else
    top = 0;
    bottom = round(dh+0.1);
    left = 0;
    right = round(dw+0.1);
end

color=114;

if shape~=newUnpad
img = padarray(img,[top,left],color,'pre');
img = padarray(img,[bottom,right],color,'post');
else
img = padarray(image,[top,left],color,'pre');
img = padarray(img,[bottom,right],color,'post');
end

img = single(img);

% Rescale image pixes to [0,1]
img = img./255;
end