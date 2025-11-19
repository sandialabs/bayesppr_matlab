classdef bpprBirthProposal
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
        PtP
        BtP
        Pty
        idx_ridge
        basis_idx
        qf_info
        log_mh
        log_mh_bd
        sse
    end

    methods
        function obj = bpprBirthProposal(state, data, prior, specs)
            obj.n_ridge = state.n_ridge + 1;

            obj.n_act = randsample(prior.n_act_max, 1, true, state.w_n_act_norm);
            if specs.adapt_act_feat
                state.log_mh_act_feat = -(log(prior.n_act_max) + log(state.w_n_act_norm(obj.n_act)));

                if obj.n_act == 1
                    obj.feat = randsample(data.p,1);
                else
                    obj.feat = datasample(1:data.p, obj.n_act, 'Replace', false, 'Weights', state.w_feat_norm);
                end
                if obj.n_act > 1 && obj.n_act < prior.n_act_max
                    state.log_mh_act_feat = state.log_mh_act_feat - lchoose(data.p, obj.n_act) + log(dwallenius(state.w_feat_norm, obj.feat));
                end
            else
                % Propose features to include
                obj.feat = datasample(1:data.p, obj.n_act, 'Replace', false, 'Weights', state.w_feat_norm);
            end 

            obj.n_quant = state.n_quant + 1;
            if obj.n_act == 1
                obj.proj_dir = randsample([-1, 1],1);
            else
                % propose direction
                obj.proj_dir = rps(prior.proj_dir_mn{obj.n_act}, 0.0);
            end
            obj.proj = data.X_st(:, obj.feat) * obj.proj_dir;
            obj.ridge_type='cont';
            max_knot0 = quantile(obj.proj, prior.p_dat_max);
            rg_knot0 = (max_knot0 - min(obj.proj)) ./ prior.prob_relu;
            knot0 = max_knot0 - rg_knot0 .* rand();
            obj.knots = [knot0, quantile(obj.proj(obj.proj > knot0), prior.knot_quants)];
            if length(unique(obj.knots)) < length(obj.knots)  % duplicates
                obj.ridge_basis = nan;
                return;
            end
            % Get proposed basis function
            obj.ridge_basis = get_mns_basis(obj.proj, obj.knots);
            obj.n_basis = prior.df_spline;

            % inner product of proposed new basis functions
            obj.PtP = obj.ridge_basis' * obj.ridge_basis;
            obj.BtP = state.basis_mat' * obj.ridge_basis;
            obj.Pty = obj.ridge_basis' * data.y;

            obj.idx_ridge = state.n_ridge+1;
            basis_idx_start = state.basis_idx{obj.idx_ridge}(end)+1;
            obj.basis_idx = basis_idx_start:(basis_idx_start+obj.n_basis-1);
        end

        function obj = get_log_mh(obj, state, data, prior)
            obj.qf_info = qf_info(state.BtB(1:(state.n_basis_total+obj.n_basis), 1:(state.n_basis_total + obj.n_basis)), state.Bty(1:(state.n_basis_total + obj.n_basis)));
            obj.log_mh = nan;
            if ~isnan(obj.qf_info.qf)
                if obj.qf_info.qf < data.ssy
                    obj.log_mh_bd = get_log_mh_bd(obj.n_ridge, obj.n_quant, prior.n_ridge_max);

                    obj.sse = data.ssy - state.c_var_coefs .* obj.qf_info.qf;

                    % compute the acceptance probability
                    obj.log_mh = (state.log_mh_bd - obj.log_mh_bd + state.log_mh_act_feat + ...
                        -data.n/2 * (log(obj.sse) - log(state.sse)) + ...
                        log(prior.n_ridge_mean/(state.n_ridge+1)));
                    if strcmpi(prior.prior_coefs, 'zs')
                        obj.log_mh = obj.log_mh - obj.n_basis * log(state.var_coefs+1)/2;
                    else
                        obj.log_mh = obj.log_mh + log(10e-6);
                    end
                end
            end
        end

    end
end
