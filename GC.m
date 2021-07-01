% Generic Classifier
classdef GC < handle
    
    properties (GetAccess = public, SetAccess = protected)
        allIn   % all inputs
        allOut   % all outputs
        trainIn   % training inputs   
        trainOut   % training outputs
        testIn   % test inputs
        testOut   % test outputs
        split = 0.8   % train-test split (default 80% train, 20% test)
        c1Color = 'k'   % class 1 plot color (default red)
        c2Color = 'w'   % class 2 plot color (default blue)
        c1Shape = 'o'   % class 1 plot shape (default x)
        c2Shape = 's'   % class 2 plot shape (default o)
        x1Label = 'Feature 1'   % feature 1 label (default 'Feature 1')
        x2Label = 'Feature 2'   % feature 2 label (default 'Feature 2')
        gTitle = 'Scatter Plot'   % graph title (default 'Scatter Plot')
        testSet = 0.20   % test set percentage (default 20%)
        kernel = @GC.linKernel % kernel (default linear kernel)
        trained = false   % keep track of if classifer is trained
    end
    
    methods(Access = public)
        
        % constructor function
        function obj = GC(varargin)
            obj.generalConstr(nargin, varargin);
        end
        
        % gTitle setter
        function setGTitle(obj, str)
            if isa(str, 'char')
                obj.gTitle = str;
            else
                fprintf("Argument is not a string\n");
            end
        end
        
        % x1Label setter
        function setX1Label(obj, str)
            if isa(str, 'char')
                obj.x1Label = str;
            else
                fprintf("Argument is not a string\n");
            end
        end
        
        % x2Label setter
        function setX2Label(obj, str)
            if isa(str, 'char')
                obj.x2Label = str;
            else
                fprintf("Argument is not a string\n");
            end
        end
        
        % graph data
        function graphData(obj, x, holdGraph)
            if holdGraph
                hold on
            else
                figure;
            end
            tv = unique(obj.trainOut);
            c1Marker = strcat('k', obj.c1Shape);
            c2Marker = strcat('k', obj.c2Shape);
            ma = {c1Marker, c2Marker};
            fc = {obj.c1Color, obj.c2Color};
            for i = 1:length(tv)
                pos = find(obj.trainOut == tv(i));
                plot(x(pos,1), x(pos,2), ma{i}, 'markerfacecolor', fc{i});
            end
            xlabel(obj.x1Label);
            ylabel(obj.x2Label);
            x1 = x(:,1); x2 = x(:,2);
            axis([min(x1)-1 max(x1)+1 min(x2)-1 max(x2)+1]);
            title(obj.gTitle);
            if holdGraph
                hold off
            end
        end
        
        % shows error rate for training set
        function [total, correct, incorrect] = trainError(obj)
            if obj.trained
                X = obj.trainIn; Y = obj.trainOut;
                [total, correct, incorrect] = obj.error(X,Y);
            else
                fprintf("You haven't trained the classifier yet!\n");
            end
        end
        
        % shows error rate for test set
        function [total, correct, incorrect] = testError(obj)
            if obj.trained
                X = obj.testIn; Y = obj.testOut;
                [total, correct, incorrect] = obj.error(X,Y);
            else
                fprintf("You haven't trained the classifier yet!\n");
            end
        end
        
    end
    
    methods(Access = protected)
        
        % general constructor
        function generalConstr(obj, nargin, allArgs)
            if nargin == 0 && strcmp(class(obj), {'GC'})
                fprintf("Please provide training input and output\n");
            elseif nargin == 1
                rhs = allArgs{1};
                if any(strcmp(class(rhs), {'GC','KMeans','PNN','SVM'}))
                    obj.copyGCProps(rhs);
                else
                    fprintf("Please provide training output\n");
                end
            elseif nargin == 2
                obj.storeInOut(allArgs{1}, allArgs{2});
            elseif nargin == 3
                obj.storeInOut(allArgs{1}, allArgs{2});
                obj.setGTitle(allArgs{3});
            elseif nargin == 5
                obj.storeInOut(allArgs{1}, allArgs{2});
                obj.setGTitle(allArgs{3});
                obj.setX1Label(allArgs{4});
                obj.setX2Label(allArgs{5});
            end
        end
        
        % copy all GC class properties from rhs
        function copyGCProps(obj, rhs)
            obj.allIn = rhs.allIn;
            obj.allOut = rhs.allOut;
            obj.trainIn = rhs.trainIn;
            obj.trainOut = rhs.trainOut;
            obj.testIn = rhs.testIn;
            obj.testOut = rhs.testOut;
            obj.split = rhs.split;
            obj.c1Color = rhs.c1Color;
            obj.c2Color = rhs.c2Color;
            obj.c1Shape = rhs.c1Shape;
            obj.c2Shape = rhs.c2Shape;
            obj.x1Label = rhs.x1Label;
            obj.x2Label = rhs.x2Label;
            obj.gTitle = rhs.gTitle;
            obj.testSet = rhs.testSet;
            obj.trained = false;
        end
        
        % store all inputs and outputs
        function storeInOut(obj, in, out)
            ok = obj.verify(in, out);
            if ok
                obj.allIn = in;
                obj.allOut = out;
                obj.makeSplit();
            end
        end
        
        % verify validity of all input and output
        function ok = verify(obj, inputs, outputs)
            inSize = size(inputs);
            outSize = size(outputs);
            gt3Size = max(inSize) > 3 && max(outSize) > 3;
            in2out1 = min(inSize) == 2 && min(outSize) == 1;
            sameInOutSize = max(inSize) == max(outSize);
            if(gt3Size && in2out1 && sameInOutSize)
                ok = true;
            else
                ok = false;
                fprintf("Invalid input and output dimensions");
            end
        end
        
        % split all inputs and outputs into train and test set
        function makeSplit(obj)
            X = obj.allIn; Y = obj.allOut; XY = [X Y];
            trainSize = floor(length(X) * obj.split);
            testSize = length(X) - trainSize;
            XY = XY(randperm(length(XY)), :);
            obj.trainIn = XY(1:trainSize, 1:2);
            obj.trainOut = XY(1:trainSize, 3);
            obj.testIn = XY(trainSize+1:trainSize+testSize, 1:2);
            obj.testOut = XY(trainSize+1:trainSize+testSize, 3);
        end
        
    end
    
    methods(Static)
        
        % linear kernel (default)
        function k = linKernel(x1, x2)
            k = x1*x2';
        end
        
    end
    
end