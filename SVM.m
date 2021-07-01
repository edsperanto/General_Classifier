classdef SVM < GC
    
    properties
        weights   % placeholder to store solved weights [bias slope]
        alpha   % output from quadratic programming
    end
    
    methods(Access = public)
        
        function obj = SVM(varargin)
            obj.generalConstr(nargin, varargin);
            if nargin == 1
                rhs = varargin{1};
                if strcmp(class(rhs), {'SVM'})
                    obj.trained = rhs.trained;
                    obj.weights = rhs.weights;
                    obj.alpha = rhs.alpha;
                end
            end
        end
        
        % train support vector machine
        % code copied from Canvas/Code & Data/MLX Files/HMSVM.mlx
        function train(obj)
            
            x = obj.trainIn; t = obj.trainOut;
            N = size(x,1);
            K = obj.kernel(x,x);
            H = (t*t').*K + 1e-5*eye(N);
            f = ones(N,1);
            A = [];b = [];
            LB = zeros(N,1); UB = inf(N,1);
            Aeq = t';beq = 0;

            % Following line runs the SVM
            options = optimset('Display', 'off');
            obj.alpha = quadprog(H,-f,A,b,Aeq,beq,LB,UB,[],options);
            % Compute the bias
            fout = sum(repmat(obj.alpha.*t,1,N).*K,1)';
            pos = find(obj.alpha>1e-6);
            bias = mean(t(pos)-fout(pos));
            
            w = sum(repmat(obj.alpha.*t,1,2).*x,1)';
            obj.weights(1) = -bias/w(2);   % bias
            obj.weights(2) = -w(1)/w(2);   % slope
            obj.trained = true;
            
            obj.plotSVMBoundary(x);
            
        end
        
        % plot classification boundary
        function plotSVMBoundary(obj, x)
            figure;hold off 
            pos = find(obj.alpha>1e-6);
            plot(x(pos,1), x(pos,2), ...
                'ko', 'markersize', 15, ... 
                'markerfacecolor', [0.6 0.6 0.6], ...
                'markeredgecolor',[0.6 0.6 0.6]);
            obj.graphData(x, true);
            xp = xlim;
            yp = obj.weights(1) + obj.weights(2) * xp;
            hold on;
            title(strcat(obj.gTitle, " SVM Boundary"));
            plot(xp,yp,'k','linewidth',2);
            hold off;
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
        function predClass = classify(obj, x)
            w = obj.weights;
            actual = x(2);
            predX2 = w(1) + w(2) * x(1);
            if actual < predX2
                predClass = -1;
            else
                predClass = 1;
            end
        end
        
    end
    
end