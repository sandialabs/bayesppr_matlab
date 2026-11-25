classdef qf_info
    properties
        dim
        chol
        ls_est
        qf
        fullrank
        inv_chol
    end

    methods

        function obj = qf_info(BtB, Bty, intercept)
            % Get the quadratic form y'X solve(X'X) X'y,
            % as well as least squares beta and cholesky of X'X.
            % With an intercept (the default) the first pivot, the
            % intercept's, is left out of the conditioning check.
            if nargin < 3, intercept = true; end

            % qf is set to NaN on failure so that callers' ~isnan(qf) guard is
            % explicit rather than relying on the behaviour of if on an empty.
            try
                R = chol(BtB);
            catch
                obj.fullrank = false;
                obj.qf = nan;
                return
            end

            dr = diag(R);
            if length(dr) > 1
                if intercept, dcheck = dr(2:end); else, dcheck = dr; end
                if max(dcheck)/min(dr) > 1e3
                    obj.fullrank = false;
                    obj.qf = nan;
                    return
                end
            end

            tmp1 = (R')\Bty;
            bhat = R\tmp1;
            qf = Bty' * bhat;
            obj.dim = length(dr);
            obj.chol = R;
            obj.ls_est = bhat;
            obj.qf = qf;
            obj.fullrank = true;
        end

        function obj = get_inv_chol(obj)
            obj.inv_chol = obj.chol\eye(obj.dim);
        end

        function obj = update_y(obj, Bty)
            % Least squares beta and quadratic form for a new X'y, reusing
            % the Cholesky factor (and its inverse), which depend only on X.
            tmp1 = (obj.chol')\Bty;
            obj.ls_est = obj.chol\tmp1;
            obj.qf = Bty' * obj.ls_est;
        end
    end
end
