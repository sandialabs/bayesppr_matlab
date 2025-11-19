classdef bpprDeathProposal
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
        w_feat
        w_feat_norm
        idx_basis
    end

    methods
        function obj = bpprDeathProposal(state, data, prior, specs)
            obj.idx_ridge = randsample(state.n_ridge, 1);

            obj.n_act = state.n_act(obj.idx_ridge);
            obj.feat = state.feat(obj.idx_ridge, 1:obj.n_act);
            if specs.adapt_act_feat
                state.log_mh_act_feat = log(prior.n_act_max) + log((state.w_n_act(obj.n_act)-1)/(sum(state.w_n_act)-1));

                obj.w_feat = state.w_feat;
                obj.w_feat(obj.feat) = obj.w_feat(obj.feat) - 1;
                obj.w_feat_norm = obj.w_feat/sum(obj.w_feat);
                if obj.n_act > 1
                    state.log_mh_act_feat = state.log_mh_act_feat + lchoose(data.p, obj.n_act) + log(dwallenius(obj.w_feat_norm, obj.feat)); 
                end
            end

            obj.idx_basis = 1:state.n_basis_total;
            obj.idx_basis(state.basis_idx{obj.idx_ridge + 1}) = [];

            obj.n_ridge = state.n_ridge - 1;
            obj.n_quant = state.n_quant - 1;
        end

        function obj = get_log_mh(obj, state, data, prior)
            obj.qf_info = qf_info(state.BtB(obj.idx_basis, obj.idx_basis), state.Bty(obj.idx_basis));
            obj.log_mh = nan;
            if ~isnan(obj.qf_info.qf)
                if obj.qf_info.qf < data.ssy
                    obj.log_mh_bd = get_log_mh_bd(obj.n_ridge, obj.n_quant, prior.n_ridge_max);

                    obj.sse = data.ssy - state.c_var_coefs .* obj.qf_info.qf;
                    obj.n_basis = state.n_basis_ridge(obj.idx_ridge+1);

                    % compute the acceptance probability
                    obj.log_mh = (state.log_mh_bd - obj.log_mh_bd + state.log_mh_act_feat + ...
                        -data.n/2 * (log(obj.sse) - log(state.sse)) + ...
                        log(state.n_ridge/prior.n_ridge_mean));
                    if strcmpi(prior.prior_coefs, 'zs')
                        obj.log_mh = obj.log_mh + 0.5*obj.n_basis * log(state.var_coefs+1);
                    else
                        obj.log_mh = obj.log_mh - log(10e-6);
                    end
                end
            end
        end

    end
end
