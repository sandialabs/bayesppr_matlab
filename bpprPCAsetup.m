classdef bpprPCAsetup
    %  Wrapper to get principal components that would be used for bpprPCA.  Mainly used for checking how many PCs should be used.

    % y: response matrix (array) of dimension nxq, where n is the number of training examples and q is the number of multivariate/functional
    % responses
    % npc: number of principal components to use (integer, optional if percVar is specified).
    % percVar: percent (between 0 and 100) of variation to explain when choosing number of principal components
    % (if npc=None).
    % center: whether to center the responses before principal component decomposition (boolean).
    % scale: whether to scale the responses before principal component decomposition (boolean).
    % returns object with plot method.

    properties
        y
        y_mean
        y_sd
        y_scale
        newy
        evals
        basis
    end

    methods
        function obj = bpprPCAsetup(y, center, scale)
            if nargin < 2
                center = true;
                scale = false;
            elseif nargin < 3
                scale = false;
            end

            obj.y = y;
            obj.y_mean = 0;
            obj.y_sd = 1;
            if center
                obj.y_mean = mean(y, 1);
            end
            if scale
                obj.y_sd = std(y, 1);
                obj.y_sd(obj.y_sd==0) = 1;
            end
            obj.y_scale = obj.y;
            for i = 1:size(obj.y,1)
                obj.y_scale(i,:) = (obj.y(i,:) - obj.y_mean)/obj.y_sd;
            end
            [U,S,V] = svd(obj.y_scale');
            obj.evals = diag(S).^2;
            obj.basis = U * S;
            obj.newy = V';
        end

        function plot(obj, npc, percVar)
            % Plot of principal components, eigenvalues

            % * left - principal components; grey are excluded by setting of npc or percVar
            % * right - eigenvalues (squared singular values), colored according to principal components
            cs = cumsum(obj.evals)/sum(obj.evals) * 100;

            if nvargin < 2
                npc = NaN;
                percVar = NaN;
            elseif nvargin < 1
                percVar = NaN;
            end

            if isnan(npc) && percVar == 100
                npc = length(obj.evals);
            end
            if isnan(npc) && ~isnan(percVar)
                npc = find(cs >= percVar, 1) + 1;
            end
            if isnan(npc) || npc > length(obj.evals)
                npc = length(obj.evals);
            end

            figure()
            subplot(1,2,1)
            if npc < length(obj.evals)
                plot(obj.basis(:,npc:end), 'Color', [0.6, 0.6, 0.6])
            end
            hold all
            for i = 1:npc
                plot(obj.basis(:,i))
            end
            ylabel('principal components')
            xlabel('multivariate/functional index')

            subplot(1,2,2)
            x = 1:length(obj.evals)+1;
            if npc < length(obj.evals)
                scatter(x(npc:end), cs(npc:end), 'Color', [0.6, 0.6, 0.6])
            end
            hold all
            for i = 1:npc
                scatter(x(i), cs(i))
            end
            xline(npc)
            ylabel('cumulative eignevalues (percent variance)')
            xlabel('index')

        end
    end
end
