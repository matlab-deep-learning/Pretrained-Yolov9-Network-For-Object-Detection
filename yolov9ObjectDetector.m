% yolov9ObjectDetector Detect objects using YOLO v9 deep learning detector.
%
% detector = yolov9ObjectDetector(detectorName) loads a pretrained YOLO v9
% object detector specified by detectorName. detectorName must be either
% 'Yolov9t', 'Yolov9s', 'Yolov9m', 'Yolov9c', or 'Yolov9e'.
%
% Inputs:
% -------
%    detectorName   Specify the name of the pretrained YOLO v9 deep learning
%                   model as a string or character vector. The value must
%                   be one of the following:
%
%                   'Yolov9t'   Use this model for speed and efficiency.
%
%                   'Yolov9s'   Use this model for a balance between speed
%                               and accuracy, suitable for applications
%                               requiring real-time performance with good
%                               detection quality.
%
%                   'Yolov9m'   Use this model for higher accuracy with
%                               moderate computational demands.
%
%                   'Yolov9c'   Use this model to prioritize maximum
%                               detection accuracy for high-end systems, at
%                               the cost of computational intensity.
%
%                   'Yolov9e'   Use this model to get most accurate
%                               detections but requires significant
%                               computational resources, ideal for high-end
%                               systems prioritizing detection performance.
%
%
%
% yolov9ObjectDetector properties:
%   ModelName                    - Name of the trained object detector.
%   Network                      - YOLO v9 object detection network. (read-only)

%
% yolov9ObjectDetector methods:
%   detect                       - Detect objects in an image.
%
% Example 1: Detect objects using pretrained YOLO v9 detector.
% ------------------------------------------------------------
% % Load the pretrained detector.
% detector = yolov9ObjectDetector();
%
% % Read test image.
% I = imread('highway.png');
%
% % Run detector.
% [bboxes, scores, labelIds] = detect(detector, I);
%
% % Display results.
% helper.plotDetections(img_orig,bboxes,scores,labelIds);
%
% Example 2: Detect objects using 'Yolov9m' pretrained model.
% -----------------------------------------------------------
% % Load the pretrained detector.
% detector = yolov9ObjectDetector('Yolov9m');
%
% % Read test image.
% I = imread('highway.png');
%
% % Run detector.
% [bboxes, scores, labelIds] = detect(detector, I);
%
% % Display results.
% helper.plotDetections(img_orig,bboxes,scores,labelIds);

% Copyright 2024 The MathWorks, Inc.


classdef yolov9ObjectDetector < vision.internal.detector.ObjectDetector
    
    properties (SetAccess=protected)
        % Network is a dlnetwork object with image input layer.
        Network
    end
    
    methods
        function this = yolov9ObjectDetector(modelName)

            % Loads and configure the pretrained model as specified in detectorName.
            if nargin<1
            % If modelName is not specified, load medium version ('Yolov9m')
                this.Network = iGetMediumNetworkDetector();
            else
            % Load the pretrained model  
                data = downloadPretrainedYOLOv9(modelName);
                this.Network = data.net;
            end

            this.Network = initialize(this.Network);

            if nargin==1
            this.ModelName = modelName;
            else
            this.ModelName = 'Yolov9m';
            end

        end


        function [bboxes, scores, labelIds] = detect(detector,image,options)
            % detect detects objects in an image using a pre-trained Yolov9 model
            % and returns corresponding bounding boxes, confidence scores and classIds
            %
            % Inputs:
            % detector                  - Pre-trained Yolov9 detector object.
            % image                     - Input image in RGB format.
            % 
            % [...] = detect(..., Name=Value) specifies additional
            % name-value pairs described below:
            % 
            % numClasses                - Number of classes yolov9Detector is trained on.
            % executionEnvironment      - Specifies the execution environment ('auto', 'gpu', 'cpu').
            %
            % Outputs:
            % bboxes                    - Bounding boxes of detected objects, each row is [x, y, width, height].
            % scores                    - Confidence scores for each detected object.
            % labelIds                  - Class IDs for each detected object.

            % Copyright 2024 The MathWorks, Inc


            arguments
                detector yolov9ObjectDetector
                image {mustBeA(image,["numeric","gpuArray"]),mustBeNonempty} 
                options.numClasses (1,1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger, mustBeFinite, mustBeNonsparse, mustBeNonempty} = 80
                options.executionEnvironment {mustBeMember(options.executionEnvironment,{'gpu','cpu','auto'})} = "auto"
            end

            % Pre-process Image
            inputSize = detector.Network.Layers(1).InputSize;
            imgp = preprocess(detector,image,inputSize(1:2));

            % Convert image to dlarray
            dlArray = dlarray(double(imgp), "SSCB");

            % Use GPU for computation if specified and available
            if (options.executionEnvironment == "auto" && canUseGPU) || options.executionEnvironment == "gpu"
                dlArray = gpuArray(dlArray);
            end

            % Perform prediction using the provided network and dlArray
            outputFeatures = predict(detector,dlArray);
            
            out1 = outputFeatures{1};
            out2 = outputFeatures{2};
            out3 = outputFeatures{3};

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
      
            dim1 = size(out, 1);
            dim2 = size(out, 2);

            endIndex = dim2 - options.numClasses;

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

            %Constant 1
            ConstantDlArray1 = dlarray(single(helper.generateCenters(sizes))); %Anchor boxes centres

            m1 = min((size(imgp, 1) / size(out1, 1)), (size(imgp, 2) / size(out1, 2)));
            m2 = min((size(imgp, 1) / size(out2, 1)), (size(imgp, 2) / size(out2, 2)));
            m3 = min((size(imgp, 1) / size(out3, 1)), (size(imgp, 2) / size(out3, 2)));
            Array2 = [repmat(m1, size(out1, 1)*size(out1, 2), 1); repmat(m2, size(out2, 1)*size(out2, 2), 1); repmat(m3, size(out3, 1)*size(out3, 2), 1)];

            % Constant 2
            ConstantDlArray2 = dlarray(single(Array2)); %Strides

            %Onnx post-processing operations
            sub1 = ConstantDlArray1 - split1;
            add1 = ConstantDlArray1 + split2;
            add3 = sub1 + add1;
            div1 = add3 ./ 2;
            sub2 = add1 - sub1;
            concat1 = cat(2, div1, sub2);
            mul1 = concat1.*ConstantDlArray2;
            combout = cat(2, mul1, splitout2);

            %Get the final feature map
            featureMap = permute(combout, [2, 1]);
            %finalout = reshape(final, [1, size(final, 1), size(final, 2)]);

            %Post-process obtained feature map
            output = postprocess(detector,featureMap,imgp, image);

            %Obtain bounding boxes, confidence scores and classIds
            bboxes = output(:,1:4);
            scores = output(:,5);
            labelIds = output(:,6);
           
        end
    end
    %----------------------------------------------------------------------
    methods(Hidden)
        %------------------------------------------------------------------
        % Preprocess input data.
        %------------------------------------------------------------------
        function img = preprocess(detector,original_image,inputLayerSize)
            % preprocess prepares an image for Yolov9 detection by adjusting its size and channel order.
            %
            % Inputs:
            % detector         - Pre-trained Yolov9 detector object.
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

        %------------------------------------------------------------------
        % Predict output feature maps.
        %------------------------------------------------------------------
        function outputFeatures = predict(detector,dlX)
            % This method predicts features of the preprocessed image dlX.
            % The outputFeatures is a N-by-1 cell array, where N are the
            % number of outputs in network. Each cell of outputFeature
            % contains predictions from an output layer.

            network = detector.Network;
            outputFeatures = cell(length(network.OutputNames), 1);
            [outputFeatures{:}] = predict(network, dlX);

        end

        %------------------------------------------------------------------
        % Postprocess output feature maps.
        %------------------------------------------------------------------
        function final_output = postprocess(detector,featureMap, preprocessed_image, original_image)
            % postprocess processes the feature map from YOLOv9 detection and returns final bounding boxes, confidence scores, and class IDs.
            %
            % Inputs:
            % detector              - Pre-trained Yolov9 detector object.
            % featureMap            - The output feature map from the YOLOv9 detector.
            % preprocessed_image    - The image after being pre-processed for YOLOv9 input.
            % original_image        - The original image before any processing.
            %
            % Outputs:
            % final_output          - An array containing the final bounding boxes, confidence scores, and class IDs. Each row corresponds to a detected object.

            % Copyright 2024 The MathWorks, Inc

            % Initialization and Configuration
            conf_thresh = 0.25; %Confidence threshold
            %conf_thresh = 0.01; %Try this for Evaluation
            iou_thres = 0.70; % IoU threshold for NMS
            num_classes = 80; % Total number of classes

            % Pre-process Predictions
            pred = permute(featureMap, [2,1]);

            % Extract class scores and find max values
            class_scores = max(pred(:, 5:5+num_classes-1), [], 2);
            conf_mask = class_scores > conf_thresh; % Mask for detections above the confidence threshold
            pred = pred(conf_mask, :); %Fiter predictions based on Confidence Threshold

            box = pred(:,1:4);
            cls = pred(:,5:5+num_classes-1);

            % Convert bounding boxes from [x_center, y_center, width, height] to [x1, y1, w, h] for NMS
            box = helper.XcYcwh2xywh(box);

            % Filter out class with maximum confidence probability
            [classProbs, classIdx] = max(cls,[],2);
            scorePred = classProbs;
            classPred = classIdx;

            % Filtered detections based on confidence
            filtered_boxes = box;
            filtered_scores = scorePred;
            filtered_classes = classPred;

            filtered_scores = double(extractdata(filtered_scores));
            filtered_classes = double(extractdata(filtered_classes));


            % Apply NMS
            [selectedBboxes, selectedScores, selectedClasses] = selectStrongestBboxMulticlass(filtered_boxes, filtered_scores,filtered_classes, ...
                'RatioType', 'Union', 'OverlapThreshold', iou_thres);

            % Prepare Final Output
            % Combine NMS survived boxes, scores, and classes
            final_output = [selectedBboxes, selectedScores, selectedClasses];

            % Post-process bounding boxes to map back to original image dimensions
            final_output(:,1:4) = helper.xywh2xyxy(final_output(:,1:4)); % Converting [x,y,w,h] to [x1,y1,x2,y2] for scaling
            final_output(:,1:4) = helper.scale_boxes(size(preprocessed_image), final_output(:,1:4), size(original_image)); % Map the final bboxes to original dimensions

            % Obtain the final bounding boxes in standard [x,y,w,h] format
            final_output(:,1:4) = helper.xyxy2xywh(final_output(:,1:4));
            final_output = gather(final_output);

        end
        
    end

    %======================================================================
    % Save/Load
    %======================================================================
    methods(Hidden)
        function s = saveobj(this)
            s.Version                      = 1.0;
            s.ModelName                    = this.ModelName;
            s.Network                      = this.Network;
        end
    end

end

%--------------------------------------------------------------------------
function network = iGetMediumNetworkDetector()
% The iGetMediumNetworkDetector function loads 'Yolov9m' network 
% pretrained on COCO dataset.
%
% Copyright 2024 The MathWorks, Inc.

detectorName = "Yolov9m";
data = downloadPretrainedYOLOv9 (detectorName);
network = data.net;
end

%--------------------------------------------------------------------------
function model = downloadPretrainedYOLOv9(modelName)
% The downloadPretrainedYOLOv9 function downloads a YOLO v9 network 
% pretrained on COCO dataset.
%
% Copyright 2024 The MathWorks, Inc.

supportedNetworks = ["Yolov9t", "Yolov9s", "Yolov9m", "Yolov9c", "Yolov9e"];
validatestring(modelName, supportedNetworks);

modelName = convertContainedStringsToChars(modelName);

netMatFileFullPath = fullfile(strcat(pwd,'/models'), [modelName, '.mat']);

if ~exist(netMatFileFullPath,'file')
    fprintf(['Downloading pretrained ', modelName ,' network.\n']);
    fprintf('This can take several minutes to download...\n');
    url = ['https://insidelabs-git.mathworks.com/zkhan/pretrained-yolov9-network-for-object-detection/-/tree/main/models', modelName, '.mat'];
    websave(netMatFileFullPath, url);
    fprintf('Done.\n\n');
else
    fprintf(['Pretrained ', modelName, ' network already exists.\n\n']);
end

model = load(netMatFileFullPath);
end