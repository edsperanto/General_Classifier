classdef PNN < GC
    
    properties
        c1Coords   % store coordinates of train set for class 1
        c2Coords   % store coordinates of train set for class 2
        stdDev = [0, 0]   % to store standard deviation of x1 and x2
    end
    
    methods(Access = public)
        
        function obj = PNN(varargin)
            obj.generalConstr(nargin, varargin);
            if nargin == 1
                rhs = varargin{1};
                if strcmp(class(rhs), {'PNN'})
                    obj.trained = rhs.trained;
                    obj.c1Coords = rhs.c1Coords;
                    obj.c2Coords = rhs.c2Coords;
                    obj.stdDev = rhs.stdDev;
                end
            end
        end
        
        % train probabalistic neural network
        function train(obj)
            % store training coordinates
            m = length(obj.trainIn);
            c1Pos = find(obj.trainOut == -1);
            c2Pos = find(obj.trainOut == 1);
            obj.c1Coords = zeros(length(c1Pos), 2);
            obj.c2Coords = zeros(length(c2Pos), 2);
            for i=1:length(obj.c1Coords)
                obj.c1Coords(i, 1:2) = obj.trainIn(c1Pos(i), 1:2);
            end
            for i=1:length(obj.c2Coords)
                obj.c2Coords(i, 1:2) = obj.trainIn(c2Pos(i), 1:2);
            end
            % store standard deviations
            x1 = obj.trainIn(:,1);
            x2 = obj.trainIn(:,2);
            % obj.stdDev(1) = (max(x1)-min(x1))/m*3;
            % obj.stdDev(2) = (max(x2)-min(x2))/m*3;
            obj.stdDev(1) = 1;
            obj.stdDev(2) = 0.4;
            % graph contour for each class
            obj.plotPNNContour(obj.trainIn);
            obj.trained = true;
        end
        
        % plot PNN distribution contour
        function plotPNNContour(obj, x)
            C1 = @(x,y) obj.classProb(obj.c1Coords, x, y, obj.stdDev);
            C2 = @(x,y) obj.classProb(obj.c2Coords, x, y, obj.stdDev);
            figure; fcontour(C1, 'LineColor', 'r');
            hold on; fcontour(C2, 'LineColor', 'b');
            obj.graphData(x, true);
            title(strcat(obj.gTitle, " PNN Contours"));
        end
        
        % shows error rate
        function [total, correct, incorrect] = error(obj, x, actual)
            correct = 0; incorrect = 0; total = length(x);
            for i = 1:total
                pred = obj.classify(x(i,:));
                if actual(i) == pred
                    correct = correct + 1;
                else
                    incorrect = incorrect + 1;
                end
            end
        end
        
        % classify a new data point
        function predClass = classify(obj, input)
            x1 = input(1); x2 = input(2);
            c1Prob = obj.classProb(obj.c1Coords, x1, x2, obj.stdDev);
            c2Prob = obj.classProb(obj.c2Coords, x1, x2, obj.stdDev);
            if c1Prob > c2Prob
                predClass = -1;
            else
                predClass = 1;
            end
        end
        
    end
    
    methods(Access = protected)
        
        function z = gauss2D(obj, x, y, x0, y0, sdX, sdY)
            xComp = (x-x0)^2/(2*sdX^2);
            yComp = (y-y0)^2/(2*sdY^2);
            z = exp(-(xComp + yComp));
        end
        
        function clProb = classProb(obj, coords, x, y, stdDev)
            sd1 = stdDev(1); sd2 = stdDev(2); clProb = 0;
            for i = 1:length(coords)
                x10 = coords(i, 1); x20 = coords(i, 2);
                clProb = clProb + obj.gauss2D(x, y, x10, x20, sd1, sd2);
            end
        end
        
    end
    
end