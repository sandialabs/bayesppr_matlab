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

        function obj = qf_info(BtB, Bty)
            % Get the quadratic form y'X solve(X'X) X'y,
            % as well as least squares beta and cholesky of X'X

            try
                R = chol(BtB);
            catch
                obj.fullrank = false;
                return
            end

            dr = diag(R);
            if length(dr) > 1
                if max(dr(2:end))/min(dr) > 1e3
                    obj.fullrank = false;
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
    end
end
