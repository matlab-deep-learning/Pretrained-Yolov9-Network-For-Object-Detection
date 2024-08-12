function yolov9Net = importYOLOv9Model(modelPath)
% importYOLOv9Model converts the '.onnx' model file to MATLAB dlNetwork,
% replacing all the onnx custom layers with MATLAB supported layers
% for various variants of YOLOv9
%
% Inputs:
% modelPath           -  root directory containing 'yolov9.onnx' file
%        
% Outputs:
% yolov9Net           -  Yolov9 dlNetwork free from onnx dependencies

% Copyright 2024 The MathWorks, Inc

file= dir(fullfile(modelPath,"*.onnx"));
networkImported = importNetworkFromONNX(file(1).name);

% Obtain Layer Names
info = analyzeNetwork(networkImported,Plots="none");
layerNames = info.LayerInfo.Name;
connections = networkImported.Connections;

% Find index of first non-essential layer that should be
% removed
remLayersIdx = find(contains(layerNames,'shape_To_Transpose','IgnoreCase',true));

% Remove other non-essential layers
idx = 1;
numLayersToRemove = size(layerNames,1)-remLayersIdx+1;
layersToBeRemoved = cell(1, numLayersToRemove);
for i = remLayersIdx:size(layerNames,1)
    layersToBeRemoved{1,idx} = layerNames{i,1};
    idx = idx + 1;
end


% Find indices of split layers that should be replaced
% with custom splitLayer
splitLayerIdx = find(contains(layerNames,'SliceLayer'));
flag=0;
for k = 1:height(connections)
    if contains(string(connections.Source{k}),layerNames(splitLayerIdx(1)))
        if contains(string(connections.Source{k}),"ou")
            flag=1;
            break;
        else
            break;
        end
    end
end
for i = 1:numel(splitLayerIdx)
    % Create custom split layer
    layer1 = CustomSliceLayer(['Slice_Layer',num2str(i)],2);
    % Replace ONNX Split Layer with custom split layer
    if(flag==1)
    for k = 1:height(connections)
        if strcmp(string(connections.Destination{k}),layerNames(splitLayerIdx(i)))
            inputSliceLayerName = connections.Source(k);
            break;
        end
    end
    
    SliceOut1={};
    for j = 1:height(connections)
       if contains(string(connections.Source{j}),layerNames(splitLayerIdx(i))) 
        if contains(connections.Source{j},'out')
            SliceOut2 = connections.Destination(j);
        else
            SliceOut1(end+1)= connections.Destination(j);
        end
       end 
    end

    networkImported = addLayers(networkImported,layer1);
    networkImported = removeLayers(networkImported,layerNames(splitLayerIdx(i)));
    networkImported = connectLayers(networkImported,inputSliceLayerName{1},layer1.Name);
    networkImported = connectLayers(networkImported,[layer1.Name,'/out2'],SliceOut2{1});
    
    for z=1:length(SliceOut1)
    networkImported = connectLayers(networkImported,[layer1.Name,'/out1'],SliceOut1{z});    
    end
    
    end

    if(flag==0)     
        networkImported = replaceLayer(networkImported,layerNames(splitLayerIdx(i)),layer1,ReconnectBy='order');
    end
end  

% Find indices of resize layers that should be replaced
% with resize2D Layer
resizeLayerIdx = find(contains(layerNames,'ResizeLayer'));
for i = 1:numel(resizeLayerIdx)
    
    % Create resize2D layer
    layerResize = resize2dLayer('EnableReferenceInput',true, 'Name', ['Resize_Layer',num2str(i)],"Method","nearest","GeometricTransformMode","asymmetric","NearestRoundingMode","floor");
    
    % Replace ONNX Resize Layer with resize2D layer
    
    nextLayerName = layerNames(resizeLayerIdx(i)+1);
    
    for k = 1:height(connections)
        if strcmp(string(connections.Destination{k}),strcat(nextLayerName,'/in2'))
            input1ConcatLayerName = connections.Source(k);
            break;
        end
    end

    for j = 1:height(connections)
        if strcmp(string(connections.Destination{j}),layerNames(resizeLayerIdx(i)))
            inputResizeLayerName = connections.Source(j);
            break;
        end
    end

    networkImported = addLayers(networkImported,layerResize);
    networkImported = connectLayers(networkImported,input1ConcatLayerName{1},strcat(layerResize.Name,'/ref'));
    networkImported = connectLayers(networkImported,inputResizeLayerName{1},strcat(layerResize.Name,'/in'));
    networkImported = disconnectLayers(networkImported,layerNames(resizeLayerIdx(i)),strcat(nextLayerName,'/in1'));
    networkImported = connectLayers(networkImported,layerResize.Name,strcat(nextLayerName,'/in1'));     
end

for i=1:numel(resizeLayerIdx)
networkImported = removeLayers(networkImported,layerNames(resizeLayerIdx(i)));
end

% Check if any paddingLayers and remove them
if (find(contains(layerNames,'PadLayer','IgnoreCase',true)))
PadLayersIdx = find(contains(layerNames,'PadLayer','IgnoreCase',true));

for i = 1:numel(PadLayersIdx)
        
    for j = 1:height(connections)
        if strcmp(connections.Destination(j),layerNames(PadLayersIdx(i)))
            inputPadLayerName = connections.Source(j);
            break;
        end
    end
    
    for k = 1:height(connections)
        if strcmp(connections.Source(k),layerNames(PadLayersIdx(i)))
            outputPadLayerName = connections.Destination(k);
            break;
        end
    end
    networkImported = removeLayers(networkImported,layerNames(PadLayersIdx(i)));
    networkImported = connectLayers(networkImported,inputPadLayerName{1},outputPadLayerName{1});
end
end

% Check if any Unsqueeze_To_ReduceSumLayer and replace it with custom resizeConcatSumLayer layer
if (find(contains(layerNames,'Unsqueeze_To_ReduceSumLayer','IgnoreCase',true)))
ReduceSumLayersIdx = find(contains(layerNames,'Unsqueeze_To_ReduceSumLayer','IgnoreCase',true));
inputReduceSumLayerName1 = {};
for i = 1:numel(ReduceSumLayersIdx)
    name = ['resizeConcatSumLayer',num2str(i)];
    resizeConcatSumLayer = ResizeConcatSumLayer(name,6,i-1);
    if i==1
     for j = 1:height(connections)
        dest = connections.Destination(j);   
        if contains(string(dest{1}),layerNames(ReduceSumLayersIdx(i)))
            inputReduceSumLayerName1{end+1} = connections.Source(j);
        end
     end
    end
    
    if (i>1)
    for j = 1:height(connections)
        dest = connections.Destination(j);
        if contains(string(dest{1}),strcat(layerNames(ReduceSumLayersIdx(i)),'/in1'))
            inputReduceSumLayerName2 = connections.Source(j);
            break;
        end
    end
    end
    
    for j = 1:height(connections)
        source = connections.Source(j);
        if contains(string(source{1}),layerNames(ReduceSumLayersIdx(i)))
            OutReduceSumLayerName = connections.Destination(j);
            break;
        end
    end
    
    networkImported = removeLayers(networkImported,layerNames(ReduceSumLayersIdx(i)));
    networkImported = addLayers(networkImported,resizeConcatSumLayer);
    networkImported = connectLayers(networkImported,inputReduceSumLayerName1{1}{1},[resizeConcatSumLayer.Name,'/in1']);
    networkImported = connectLayers(networkImported,inputReduceSumLayerName1{2}{1},[resizeConcatSumLayer.Name,'/in2']);
    networkImported = connectLayers(networkImported,inputReduceSumLayerName1{3}{1},[resizeConcatSumLayer.Name,'/in3']);
    networkImported = connectLayers(networkImported,inputReduceSumLayerName1{4}{1},[resizeConcatSumLayer.Name,'/in4']);
    networkImported = connectLayers(networkImported,inputReduceSumLayerName1{5}{1},[resizeConcatSumLayer.Name,'/in5']);
    
    if i==1
        networkImported = connectLayers(networkImported,inputReduceSumLayerName1{6}{1},[resizeConcatSumLayer.Name,'/in6']);
    end
    
    if i>1
        networkImported = connectLayers(networkImported,inputReduceSumLayerName2{1},[resizeConcatSumLayer.Name,'/in6']);
    end 

    networkImported = connectLayers(networkImported,resizeConcatSumLayer.Name,OutReduceSumLayerName{1});
    
end

end

% Remove the unnecessary layers    
networkImported = removeLayers(networkImported,layersToBeRemoved);

% Find layers connected to batchSizeVerifier layer
OutToInLayers={};

for j = 1:height(connections)
    source = connections.Source(j);   
    if strcmp(string(source{1}),networkImported.Layers(2).Name)
        OutToInLayers{end+1} = connections.Destination(j);
    end
end

% Remove batchSizeVerifier layer
networkImported = removeLayers(networkImported,networkImported.Layers(2).Name);

% Connect the unconnected layers to input layer
for k=1:length(OutToInLayers)
networkImported = connectLayers(networkImported,networkImported.Layers(1).Name,OutToInLayers{k}{1});
end

% Incorporate Swish layers
yolov9Net = insertSwishLayers(networkImported);

end

function output = insertSwishLayers(net)

layers = net.Layers;
lgraph = net.layerGraph;

for ly = layers'
    if ~isa(ly,'nnet.cnn.layer.MultiplicationLayer')
        continue;
    end
    % Get the multiplication layer's name
    mulName = ly.Name;
    % Get the layers driving the multiplication layer
    [drivingLayerNames, drivingIndices] = getDrivingLayers(net,mulName);
    drivingLayers = layers(drivingIndices);
    % Check if any of them is a sigmoid layer. Proceed only if true.
    sigmoidLayerIndex = find(arrayfun(@(x) isa(x, 'nnet.cnn.layer.SigmoidLayer'), drivingLayers));
    if isempty(sigmoidLayerIndex)
        continue;
    end
    otherLayerIndex = 2 - sigmoidLayerIndex + 1;
    % Check if the sigmoid and multiply layers have the same driving layer
    sigmoidName = drivingLayerNames{sigmoidLayerIndex};
    sigmoidDrivingLayer = getDrivingLayers(net,sigmoidName);
    % If the same layer drives the sigmoid and the other input of the
    % multiplication layer, sigmoid+mutliplication layers can be replaced with
    % a swish layer.
    otherLayer = drivingLayerNames{otherLayerIndex};
    if isequal(sigmoidDrivingLayer{1},otherLayer)
        % Revome the sigmoid layer
        lgraph = removeLayers(lgraph,sigmoidName);
        % Create a swish layer
        newLayer = swishLayer("Name",mulName);
        % Get the layers succeeding the multiplication layer
        succeedingLayers = getSucceedingLayers(net, mulName);
        % Relpace the multiplication layer with the swish layer
        lgraph = removeLayers(lgraph,mulName);
        lgraph = addLayers(lgraph, newLayer);
        lgraph = connectLayers(lgraph, otherLayer, mulName);
        for i = 1:numel(succeedingLayers)
            lgraph = connectLayers(lgraph, mulName, succeedingLayers{i});
        end
    end
    output = dlnetwork(lgraph);
end
   
end

function [drivingLayers, drivingIndices] = getDrivingLayers(net, layerName)
    lg=layerGraph(net); 
    diG = extractPrivateDirectedGraph(lg);

    layerTable = diG.Nodes;
    % Extract the layer names into a cell array
    allLayerNames = {layerTable.Layers.Name}';
    % Find the index of the layer of interest: layerName
    layerIndex = find(strcmp(allLayerNames,layerName));
    % Extract the array containing node connection info
    % This is an Nx2 array. Each row is a connection with
    % the fist column containing the source and the 2nd
    % the destination.
    digNodes = diG.Edges.EndNodes;
    % Get the layer indices that drive layerName
    drivingIndices = digNodes(digNodes(:,2) == layerIndex, 1);
    % Get the driving layer names based on the indices
    drivingLayers = allLayerNames(drivingIndices);
end

function succeedingLayers = getSucceedingLayers(net, layerName)
    % Extract the connections table
    connections = net.Connections;
    % Find the indices in the Source column matching layerName.
    % Then get the names in these indices from the Destination column.
    succeedingLayers = connections.Destination(strcmp(connections.Source,layerName));
end