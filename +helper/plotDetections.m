function plotDetections(img,bboxes, scores, labelIds, varargin)
% plotDetections plots detection results on an input RGB image using class IDs.
%
% Inputs:
% img        - Input RGB image.
% bboxes     - Bounding boxes of detected objects, each row is [x, y, width, height].
% scores     - Confidence scores for each detected object.
% labelIds   - Class IDs for each detected object.
% varargin   - Variable input arguments for optional parameters including:
%       - 'LineWidth'  : Line width of bounding boxes. Default is 2.
%       - 'FontSize'   : Font size of the text. Default is 12.
%       - 'Show'       : Whether to display the image. Default is true.
%       - 'Save'       : Whether to save the image. Default is false.
%       - 'Filename'   : Filename to save the image. Default is 'results.jpg'.
%

% Copyright 2024 The MathWorks, Inc.

% Default options
opts = struct('LineWidth', 2, 'FontSize', 12, 'Show', true, 'Save', false, 'Filename', 'results.jpg');
optNames = fieldnames(opts);

% Override default options with varargin
nArgs = length(varargin);
if round(nArgs/2)~=nArgs/2
   error('plotDetections needs propertyName/propertyValue pairs')
end
for pair = reshape(varargin,2,[]) %# for each pair
   if any(strcmp(pair{1}, optNames))
      opts.(pair{1}) = pair{2};
   else
      error('%s is not a recognized parameter name',pair{1})
   end
end

% Prepare for plotting
classNames = helper.getCOCOClassNames();
numClasses = length(classNames);
boxColors = hsv(numClasses);

% Plot detections
figure; imshow(img); hold on;
for i = 1:size(bboxes, 1)
    x1 = bboxes(i,1);
    y1 = bboxes(i,2);
    width = bboxes(i,3);
    height = bboxes(i,4);
    score = scores(i,1);
    classID = labelIds(i,1);
    className = classNames{classID};
    
    % Determine box and text properties
    boxColor = boxColors(classID, :);
    labelColor = 'white';
    
    % Plot bounding box and text label
    rectangle('Position', [x1, y1, width, height], 'EdgeColor', boxColor, 'LineWidth', opts.LineWidth);
    textStr = sprintf('%s: %.2f', className, score);
    text(x1, y1 - 10, textStr, 'Color', labelColor, 'FontSize', opts.FontSize,'FontWeight', 'bold','BackgroundColor', boxColor);

end
hold off;

% Show or save the image based on input options
if opts.Show
    figure(gcf);  % Bring the figure to front
end
if opts.Save
    saveas(gcf, opts.Filename); % Save the figure
end

end

