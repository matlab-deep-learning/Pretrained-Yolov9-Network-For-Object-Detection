classdef ResizeConcatSumLayer < nnet.layer.Layer ...
        & nnet.layer.Formattable ... 
        & nnet.layer.Acceleratable
    properties
        SegmentNum;
    end
    methods
        function layer = ResizeConcatSumLayer(name,numInputs,SegmentNum)
            % Set layer properties
            layer.Name = name;
            layer.NumInputs = numInputs;
            layer.SegmentNum = SegmentNum;
            % Set layer description
            layer.Description = "Split along channels, resize to max spatial dimensions, concatenate along 5th dimension, and sum along 5th dimension";
        end

   function Z = predict(layer,varargin)
            % Inputs should in order

            % Number of inputs 
            numInputs = numel(varargin);
            k = layer.SegmentNum;
            
            % Split each input along the channel dimension and take the corresponding segment
            segment = cell(1, numInputs-k);
           
            % Assign the additional layer at the corresponding index
            segment{end} = varargin{end};
           
            segmentSize = 64; 

            % Split Inputs and populate each segments
            for i = 1:numel(segment)-1
                input = varargin{i+k};
                if k==0
                   segment{i} = input(:, :, k*segmentSize+1:(2^k)*segmentSize,:);
                else
                   segment{i} = input(:, :, ((2^k)-1)*segmentSize+1:((2^(k+1)) - 1)*segmentSize,:); 
            
                end
            end

            % Determine the maximum spatial dimensions
            maxHeight = max(cellfun(@(x) size(x, 1), segment));
            maxWidth = max(cellfun(@(x) size(x, 2), segment));           
         
            % Resize spatial dimensions to [maxHeight,maxWidth] 
            for i = 1:numel(segment)-1
                    segment{i} = dlresize(segment{i},'OutputSize',[maxHeight,maxWidth],'Method','nearest','GeometricTransformMode', 'asymmetric','NearestRoundingMode','floor');           
            end 

            % Concatenate the resized segments along the 5th dimension
            if k==0
            concat = cat(5, segment{1},segment{2},segment{3},segment{4},segment{5},segment{6});
            elseif k==1
            concat = cat(5, segment{1},segment{2},segment{3},segment{4},segment{5});
            elseif k==2
            concat = cat(5, segment{1},segment{2},segment{3},segment{4});
            elseif k==3
            concat = cat(5, segment{1},segment{2},segment{3});
            else
            concat = cat(5, segment{1},segment{2});
            end
            
            % Sum along the 5th dimension
            Z = sum(concat, 5);
       
        end
    end
end


