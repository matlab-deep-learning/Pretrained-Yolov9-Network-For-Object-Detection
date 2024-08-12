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