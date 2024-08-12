% Demo for one image

%Load image
img_orig = imread("inputTeam.jpg");

% If image is grayscale
sz = size(img_orig);
if numel(sz) == 2
    img_orig = cat(3,img_orig,img_orig,img_orig);
end

%Load a pre-trained Yolov9 object detector
detector = helper.yolov9Network("Yolov9m");

%Set execution environment {"auto","gpu","cpu"}
executionEnvironment = "auto";

%Generate Predictions
[bboxes, scores, labelIds] = helper.detectYolov9(detector,img_orig,executionEnvironment);

%plot Detections (You can use either of the following methods: {plotDetections,plotObjectDetections})

%helper.plotDetections(img_orig,bboxes,scores,labelIds,'LineWidth', 1.5, 'FontSize', 10,'Save',true);
helper.plotObjectDetections(img_orig,bboxes,scores,labelIds);


    