function centersArray = generateCenters(sizes)
% generateCenters generates center coordinates for anchor boxes based on feature map sizes.
%
% Inputs:
% sizes           - An Nx2 matrix, where each row represents the [height, width] of a feature map.
%
% Outputs:
% centersArray    - An Mx2 matrix containing the [x, y] coordinates of each center point
%                   for all the anchor boxes.

% Copyright 2024 The MathWorks, Inc

% Initialize an empty array to store the centers
totalCenters = sum(prod(sizes, 2));
centersArray = zeros(totalCenters, 2);

% Index to keep track of where to insert the new centers
currentIndex = 1;

for i = 1:size(sizes, 1)
    numRows = sizes(i, 1);
    numCols = sizes(i, 2);

    % Generate the center coordinates for the current anchor box
    [X, Y] = meshgrid(0.5:1:(numRows-0.5), 0.5:1:(numCols-0.5));

    % Flatten the X and Y coordinates and combine them into a 2-column array
    centers = [Y(:), X(:)];

    % Calculate the number of centers generated in this iteration
    numCenters = size(centers, 1);

    % Insert the new centers into the pre-allocated array
    centersArray(currentIndex:(currentIndex + numCenters - 1), :) = centers;

    % Update the currentIndex for the next insertion
    currentIndex = currentIndex + numCenters;

end

end