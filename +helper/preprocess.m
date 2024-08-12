function img = preprocess(original_image,inputLayerSize)
% preprocess prepares an image for Yolov9 detection by adjusting its size and channel order.
%
% Inputs:
% original_image   - The original RGB image to be preprocessed.
% inputLayerSize   - The size of the input layer of the Yolov9 model, specified as [height, width].
%
% Outputs:
% img              - The preprocessed image ready for Yolov9 detection.

% Copyright 2024 The MathWorks, Inc.

% Switch the first and third channels to convert from rgb to bgr
img_switched = original_image(:,:,[3,2,1]);

% Apply letterBox to adjust the image size while keeping the aspect ratio
img = helper.letterBox(img_switched,auto=true,newShape=inputLayerSize);

% convert image back to rgb
img = img(:,:,[3,2,1]);

% Reshape and permute the image to prepare for Yolov9 detector
img = reshape(img, [1, size(img, 1), size(img, 2), size(img, 3)]);
img = permute(img, [2, 3, 4, 1]);

end