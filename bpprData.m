classdef bpprData
    % Structure to store data

    properties
        X
        X_st
        X_st_new
        y
        ssy
        n
        p
        mn_X
        sd_X
        feat_type
    end

    methods
        function obj = bpprData(X, y)
            obj.X = X;
            obj.y = y;
        end

        function obj = summarize(obj, prior)
            obj.n = size(obj.X,1);
            obj.p = size(obj.X,2);
            obj.ssy = obj.y' * obj.y;
            obj.mn_X = zeros(obj.p,1);
            obj.sd_X = ones(obj.p,1);
            % Classify each feature. Constant ("") and binary ("cat") features are
            % left unstandardized (mn = 0, sd = 1) so that a zero standard deviation
            % never reaches the divisor.
            obj.feat_type = strings(obj.p,1);
            for j = 1:obj.p
                n_unique = numel(unique(obj.X(:,j)));
                if n_unique == 1
                    obj.feat_type(j) = "";
                elseif n_unique == 2
                    obj.feat_type(j) = "cat";
                else
                    obj.mn_X(j) = mean(obj.X(:,j));
                    obj.sd_X(j) = std(obj.X(:,j));
                    if n_unique <= prior.df_spline
                        obj.feat_type(j) = "disc";
                    else
                        obj.feat_type(j) = "cont";
                    end
                end
            end
            obj = obj.standardize();
        end

        function obj = standardize(obj, X)
            % With one argument, standardize obj.X into obj.X_st.
            % With two, standardize X into obj.X_st_new.
            if nargin < 2
                obj.X_st = obj.X;
                for j = 1:obj.p
                    obj.X_st(:, j) = (obj.X(:, j) - obj.mn_X(j)) ./ obj.sd_X(j);
                end
            else
                obj.X_st_new = X;
                for j = 1:obj.p
                    obj.X_st_new(:, j) = (X(:, j) - obj.mn_X(j)) ./ obj.sd_X(j);
                end
            end
        end

    end
end
