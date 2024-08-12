function [bboxes,scores,labelIds] = yolov9Predict(image)

% Copyright 2024 The MathWorks, Inc.

% Load pretrained network.
persistent net
if isempty(net)
    net = coder.loadDeepLearningNetwork('Yolov9m.mat');
end

% Get the input size of the network.
inputSize = [640 640 3];

imgp = helper.preprocess(image,inputSize(1:2));

% Convert image to dlarray
dlArray = dlarray(single(imgp), "SSCB");

% Perform prediction using the provided network and dlArray
[out1, out2, out3] = predict(net,dlArray);

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
nclasses = 80;
dim1 = size(out, 1);
dim2 = size(out, 2);

endIndex = dim2 - nclasses;

splitout1 = out(:, 1:endIndex);
splitout2 = out(:, endIndex+1:dim2);
splitout2 = sigmoid(splitout2);

splitout1 = reshape(splitout1, [dim1, 16, 4]);
splitout1 = permute(splitout1, [1, 3, 2]);
splitout1 = permute(splitout1, [2, 1, 3]);
splitout1 = dlarray(single(splitout1), 'SSCB');

softmaxout = softmax(splitout1);

weights = dlarray(single(reshape(0:15, [1, 1, 16])));
bias = dlarray(single(0));

convout = dlconv(softmaxout, weights, bias);

splitoutFc = extractdata(permute(convout, [2, 1, 3, 4]));
splitoutFc = splitoutFc';
split1 = single(splitoutFc(:, 1:2));
split2 = single(splitoutFc(:, 3:4));

% Calculate constants for ONNX post-processing
sizes = [size(out1, 1) size(out1, 2); size(out2, 1) size(out2, 2); size(out3, 1) size(out3, 2)];

% Constant 1
ConstantArray1 = single(helper.generateCenters(sizes)); % Anchor boxes centres

m1 = min((size(imgp, 1) / size(out1, 1)), (size(imgp, 2) / size(out1, 2)));
m2 = min((size(imgp, 1) / size(out2, 1)), (size(imgp, 2) / size(out2, 2)));
m3 = min((size(imgp, 1) / size(out3, 1)), (size(imgp, 2) / size(out3, 2)));
Array2 = [repmat(m1, size(out1, 1)*size(out1, 2), 1); repmat(m2, size(out2, 1)*size(out2, 2), 1); repmat(m3, size(out3, 1)*size(out3, 2), 1)];

% Constant 2
ConstantArray2 = single(Array2); % Strides

% Onnx post-processing operations
sub1 = ConstantArray1 - split1;
add1 = ConstantArray1 + split2;
add3 = sub1 + add1;
div1 = add3 ./ 2;
sub2 = add1 - sub1;
concat1 = cat(2, div1, sub2);
mul1 = concat1 .* ConstantArray2;
mul1 = dlarray(mul1);

combout = cat(2, mul1, splitout2);

% Get the final feature map
featureMap = permute(combout, [2, 1]);

% Post-process obtained feature map
output = helper.postprocess(featureMap,imgp, image);

% Obtain bounding boxes, confidence scores and classIds
bboxes = output(:,1:4);
scores = output(:,5);
labelIds = output(:,6);
end
