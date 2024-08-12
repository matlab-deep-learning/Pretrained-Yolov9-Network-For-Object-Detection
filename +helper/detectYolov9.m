function [bboxes, scores, labelIds] = detectYolov9(detector,image,executionEnvironment)
% detectYolov9 detects objects in an image using a pre-trained Yolov9 model
% and returns corresponding bounding boxes, confidence scores and classIds
% 
% Inputs:
% detector                  - Pre-trained Yolov9 detector object.
% image                     - Input image in RGB format.
% executionEnvironment      - Specifies the execution environment ('auto', 'gpu', 'cpu').
% 
% Outputs:
% bboxes                    - Bounding boxes of detected objects, each row is [x, y, width, height].
% scores                    - Confidence scores for each detected object.
% labelIds                  - Class IDs for each detected object.

% Copyright 2024 The MathWorks, Inc

% Pre-process Image
inputSize = detector.Layers(1).InputSize;
imgp = helper.preprocess(image,inputSize(1:2));

% Convert image to dlarray
dlArray = dlarray(double(imgp), "SSCB");

% Default execution environment
if nargin < 3 || isempty(executionEnvironment)
    executionEnvironment = "auto"; % Default value
end

% Use GPU for computation if specified and available
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlArray = gpuArray(dlArray);
end

% Perform prediction using the provided network and dlArray
[out1, out2, out3] = predict(detector,dlArray);

% Post-process the network outputs

% Permute and reshape outputs to prepare for further processing
out1p = permute(out1, [2 1 3 4]);
out2p = permute(out2, [2 1 3 4]);
out3p = permute(out3, [2 1 3 4]);

out1r = reshape(out1p, [size(out1p, 1)*size(out1p, 2), size(out1p, 3)]);
out2r = reshape(out2p, [size(out2p, 1)*size(out2p, 2), size(out2p, 3)]);
out3r = reshape(out3p, [size(out3p, 1)*size(out3p, 2), size(out3p, 3)]);
out = cat(1, out1r, out2r, out3r);

% Post-processing ONNX Layers implementation in MATLAB
% This is specific to Yolov9c model
nclasses = 80;
dim1 = size(out, 1);
dim2 = size(out, 2);

endIndex = dim2 - nclasses;

splitout1 = out(:, 1:endIndex);
splitout2 = out(:, endIndex+1:dim2);
splitout2 = sigmoid(splitout2);

% Hard-coded values for 144 channels
splitout1 = reshape(splitout1, [dim1, 16, 4]);
splitout1 = permute(splitout1, [1, 3, 2]);
splitout1 = permute(splitout1, [2, 1, 3]);
splitout1 = dlarray(single(splitout1), 'SSCB');

softmaxout = softmax(splitout1);

weights = dlarray(single(reshape(0:15, [1, 1, 16])));
bias = dlarray(single(0));

convout = dlconv(softmaxout, weights, bias);

splitoutFc = permute(convout, [2, 1, 3, 4]);
split1 = splitoutFc(:, 1:2);
split2 = splitoutFc(:, 3:4);

% Calculate constants for ONNX post-processing
sizes = [size(out1, 1) size(out1, 2); size(out2, 1) size(out2, 2); size(out3, 1) size(out3, 2)];

% Constant 1
ConstantDlArray1 = dlarray(single(helper.generateCenters(sizes))); %Anchor boxes centres

m1 = min((size(imgp, 1) / size(out1, 1)), (size(imgp, 2) / size(out1, 2)));
m2 = min((size(imgp, 1) / size(out2, 1)), (size(imgp, 2) / size(out2, 2)));
m3 = min((size(imgp, 1) / size(out3, 1)), (size(imgp, 2) / size(out3, 2)));
Array2 = [repmat(m1, size(out1, 1)*size(out1, 2), 1); repmat(m2, size(out2, 1)*size(out2, 2), 1); repmat(m3, size(out3, 1)*size(out3, 2), 1)];

% Constant 2
ConstantDlArray2 = dlarray(single(Array2)); %Strides

% Onnx post-processing operations
sub1 = ConstantDlArray1 - split1;
add1 = ConstantDlArray1 + split2;
add3 = sub1 + add1;
div1 = add3 ./ 2;
sub2 = add1 - sub1;
concat1 = cat(2, div1, sub2);
mul1 = concat1 .* ConstantDlArray2;

combout = cat(2, mul1, splitout2);

% Get the final feature map
featureMap = permute(combout, [2, 1]);
%finalout = reshape(final, [1, size(final, 1), size(final, 2)]);

% Post-process obtained feature map
output = helper.postprocess(featureMap,imgp, image);

% Obtain bounding boxes, confidence scores and classIds
bboxes = output(:,1:4);
scores = output(:,5);
labelIds = output(:,6);

end