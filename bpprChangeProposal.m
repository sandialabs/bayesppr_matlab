classdef bpprChangeProposal
    % Structure to store data

    properties
        n_ridge
        n_act
        feat
        n_quant
        proj_dir
        proj
        ridge_type
        knots
        ridge_basis
        n_basis
        BtB
        Bty
        idx_ridge
        basis_idx
        qf_info
        log_mh
        log_mh_bd
        sse
        w_feat
        wfeat_norm
    end

    methods
        function obj = bpprChangeProposal(state, data, prior, specs)
            obj.idx_ridge = randsample(state.idx_ridge_quant, 1);

            obj.n_act = state.n_act(obj.idx_ridge);
            obj.feat = state.feat(obj.idx_ridge, 1:obj.n_act);
            proj_dir_curr = state.proj_dir(obj.idx_ridge,1:obj.n_act);

            if state.n_act(obj.idx_ridge) == 1
                obj.proj_dir = randsample([-1, 1], 1);
            else
                obj.proj_dir = rps(proj_dir_curr, specs.proj_dir_prop_prec);
            end

            obj.proj = data.X_st(:,obj.feat) * obj.proj_dir;  % get proposed projections
            
            max_knot0 = quantile(obj.proj, prior.p_dat_max);
            rg_knot0 = (max_knot0 - min(obj.proj))./prior.prob_relu;
            knot0 = max_knot0 - rg_knot0 * rand(1);
            obj.knots = [knot0, quantile(obj.proj(obj.proj > knot0), prior.knot_quants)];
            if length(unique(obj.knots)) < length(obj.knots)  % duplicates
                obj.ridge_basis = nan;
                return;
            end

            obj.ridge_basis = get_mns_basis(obj.proj, obj.knots);

            % inner product of proposed new basis functions
            PtP = obj.ridge_basis' * obj.ridge_basis;
            BtP = state.basis_mat' * obj.ridge_basis;
            Pty = obj.ridge_basis' * data.y;

            obj.BtB = state.BtB(1:state.n_basis_total, 1:state.n_basis_total);
            obj.BtB(state.basis_idx{obj.idx_ridge+1},:) = BtP';
            obj.BtB(:, state.basis_idx{obj.idx_ridge+1}) = BtP;
            obj.BtB(state.basis_idx{obj.idx_ridge+1}, state.basis_idx{obj.idx_ridge+1}) = PtP;

            obj.Bty = state.Bty(1:state.n_basis_total);
            obj.Bty(state.basis_idx{obj.idx_ridge+1}) = Pty;
        end

        function obj = get_log_mh(obj, state, data, prior)
            obj.qf_info = qf_info(obj.BtB, obj.Bty);
            obj.log_mh = nan;
            if ~isnan(obj.qf_info.qf)
                if obj.qf_info.qf < data.ssy
                    obj.sse = data.ssy - state.c_var_coefs .* obj.qf_info.qf;
                    % compute the acceptance probability
                    obj.log_mh = -data.n/2 * (log(obj.sse) - log(state.sse));
                end
            end
        end

    end
end
