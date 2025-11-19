classdef bpprState < handle
    % The current state of the RJMCMC chain, with methods for getting the
    % log posterior and for updating the state

    properties
        n_ridge
        n_act
        feat
        proj_dir
        knots
        coefs
        s2
        var_coefs
        c_var_coefs
        n_basis_ridge
        n_basis_total
        ridge_type
        idx_ridge_quant
        basis_idx
        BtB
        Bty
        qf_info
        sse
        log_mh_bd
        log_mh_feat
        idx
        phase
        w_n_act
        w_n_act_norm
        w_feat
        w_feat_norm
        n_quant
        basis_mat
        log_mh_act_feat
    end

    methods
        function obj = bpprState(data, prior, specs)
            obj.n_ridge = 0;
            obj.n_act = zeros(prior.n_ridge_max, 1);
            obj.feat = zeros(prior.n_ridge_max, prior.n_act_max);
            obj.proj_dir = zeros(prior.n_ridge_max, prior.n_act_max);
            obj.knots = zeros(prior.n_ridge_max, prior.df_spline + 2);
            obj.coefs = zeros(prior.df_spline*prior.n_ridge_max+1, 1);
            obj.coefs(1)= mean(data.y);
            obj.s2 = 1.0;

            if strcmpi(prior.prior_coefs,'zs')
                obj.var_coefs = prior.scale_var_coefs ./ prior.shape_var_coefs;
                obj.c_var_coefs = obj.var_coefs ./ (obj.var_coefs + 1);
            elseif strcmpi(prior.prior_coefs, 'flat')
                obj.var_coefs = nan;
                obj.c_var_coefs = 1.0;
            end

            % other things to track
            obj.n_basis_ridge = [1];
            obj.n_basis_total = sum(obj.n_basis_ridge);
            obj.ridge_type = [];
            obj.idx_ridge_quant = [];
            obj.n_quant = 0;
            obj.basis_mat = ones(data.n,1);
            obj.basis_idx = {1:1};
            obj.BtB = zeros(prior.n_ridge_max * prior.df_spline + 1, prior.n_ridge_max * prior.df_spline + 1);
            obj.BtB(1:obj.n_basis_total, 1:obj.n_basis_total) = obj.basis_mat' * obj.basis_mat;
            obj.Bty = zeros(prior.n_ridge_max * prior.df_spline + 1, 1);
            obj.Bty(1:obj.n_basis_total) = obj.basis_mat' * data.y;
            obj.qf_info = qf_info(obj.BtB(1:obj.n_basis_total, 1:obj.n_basis_total), obj.Bty(1:obj.n_basis_total));
            obj.qf_info = obj.qf_info.get_inv_chol();

            obj.sse = data.ssy - obj.c_var_coefs * obj.qf_info.qf;
            obj.log_mh_bd = 0.0;
            obj.log_mh_act_feat = 0.0;

            obj.idx = 1;
            if specs.n_adapt > 0
                obj.phase = 'adapt';
            elseif specs.n_burn > 0
                obj.phase = 'burn';
            else
                obj.phase = 'post-burn';
            end

            obj.w_n_act = specs.w_n_act;
            obj.w_n_act_norm = obj.w_n_act ./ sum(obj.w_n_act);
            obj.w_feat = specs.w_feat;
            obj.w_feat_norm = obj.w_feat ./ sum(obj.w_feat);
        end


        function obj = acceptBirth(obj, prop, adapt_act_feat)
            obj.n_ridge = obj.n_ridge + 1;
            obj.n_act(prop.idx_ridge) = prop.n_act;
            obj.feat(prop.idx_ridge, 1:prop.n_act) = prop.feat;
            obj.knots(prop.idx_ridge, :) = prop.knots;
            obj.proj_dir(prop.idx_ridge, 1:prop.n_act) = prop.proj_dir;

            obj.idx_ridge_quant = [obj.idx_ridge_quant, prop.idx_ridge];
            obj.n_quant = prop.n_quant;
            obj.ridge_type = [obj.ridge_type, prop.ridge_type];

            obj.basis_idx{prop.idx_ridge+1} = prop.basis_idx;
            obj.n_basis_ridge = [obj.n_basis_ridge, prop.n_basis];
            obj.n_basis_total = obj.n_basis_total + prop.n_basis;
            obj.basis_mat = [obj.basis_mat, prop.ridge_basis];

            % adapt weights
            if adapt_act_feat
                obj.w_n_act(prop.n_act) = obj.w_n_act(prop.n_act) + 1;
                obj.w_n_act_norm = obj.w_n_act/sum(obj.w_n_act);
                obj.w_feat(prop.feat) = obj.w_feat(prop.feat) + 1;
                obj.w_feat_norm = obj.w_feat / sum(obj.w_feat);
            end

            obj.qf_info = prop.qf_info;
            if ~strcmpi(obj.phase,"adapt")
                obj.qf_info = obj.qf_info.get_inv_chol();
            end
            obj.sse = prop.sse;
            obj.log_mh_bd = prop.log_mh_bd;
        end

        function obj = acceptDeath(obj, prop, adapt_act_feat)
            obj.n_basis_total = obj.n_basis_total - prop.n_basis;
            obj.n_ridge = obj.n_ridge - 1;

            obj.n_quant = obj.n_quant - 1;
            obj.idx_ridge_quant(prop.idx_ridge) = [];
            
            for k = 1:length(obj.idx_ridge_quant)
                if obj.idx_ridge_quant(k) > prop.idx_ridge
                    obj.idx_ridge_quant(k) = obj.idx_ridge_quant(k) - 1;
                end
            end

            obj.basis_mat(:, obj.basis_idx{prop.idx_ridge+1}) = [];
            obj.BtB(1:obj.n_basis_total, 1:obj.n_basis_total) = obj.BtB(prop.idx_basis, prop.idx_basis);
            obj.Bty(1:obj.n_basis_total) = obj.Bty(prop.idx_basis);

            for j = prop.idx_ridge:obj.n_ridge
                obj.n_act(j) = obj.n_act(j+1);
                obj.feat(j, :) = obj.feat(j+1, :);
                obj.knots(j, :) = obj.knots(j+1, :);
                obj.proj_dir(j, :) = obj.proj_dir(j+1, :);
                obj.basis_idx{j+1} = (obj.basis_idx{j+2}(1) - obj.n_basis_ridge(prop.idx_ridge+1)):(obj.basis_idx{j+2}(end) - obj.n_basis_ridge(prop.idx_ridge+1));
            end

            obj.basis_idx(end) = [];
            obj.n_basis_ridge(prop.idx_ridge+1) = [];

            % update weights
            if adapt_act_feat
                obj.w_feat = prop.w_feat;
                obj.w_feat_norm = prop.w_feat_norm;
                obj.w_n_act(prop.n_act) = obj.w_n_act(prop.n_act) - 1;
                obj.w_n_act_norm = obj.w_n_act./sum(obj.w_n_act);
            end

            obj.qf_info = prop.qf_info;
            if ~strcmpi(obj.phase,'adapt')
                obj.qf_info = obj.qf_info.get_inv_chol();
            end
            obj.sse = prop.sse;
            obj.log_mh_bd = prop.log_mh_bd;
        end

        function obj = acceptChange(obj, prop)
            obj.knots(prop.idx_ridge, :) = prop.knots;
            obj.proj_dir(prop.idx_ridge, 1:prop.n_act) = prop.proj_dir;
            
            obj.BtB(1:obj.n_basis_total, 1:obj.n_basis_total) = prop.BtB;
            obj.Bty(1:obj.n_basis_total) = prop.Bty;
            obj.basis_mat(:, obj.basis_idx{prop.idx_ridge+1}) = prop.ridge_basis;

            obj.qf_info = prop.qf_info;
            if ~strcmpi(obj.phase, 'adapt')
                obj.qf_info = obj.qf_info.get_inv_chol();
            end
            obj.sse = prop.sse;
        end

        function obj = sampleSDresid(obj, data)
            obj.s2 = 1./gamrnd(data.n/2, 2/obj.sse);
        end

        function obj = sampleCoefs(obj)
            obj.coefs(1:obj.n_basis_total) = (obj.c_var_coefs .* obj.qf_info.ls_est + ...
                sqrt(obj.c_var_coefs.*obj.s2) .* obj.qf_info.inv_chol * randn(obj.n_basis_total,1));
        end

        function obj = sampleVarCoefs(obj, data, prior)
            qf_comp = obj.qf_info.chol * obj.coefs(1:obj.n_basis_total);
            qf = qf_comp' * qf_comp;
            obj.var_coefs = 1/gamrnd(prior.shape_var_coefs+obj.n_basis_total/2, 1/(prior.scale_var_coefs + qf./(2*obj.s2)));
            obj.c_var_coefs = obj.var_coefs / (obj.var_coefs + 1);
            obj.sse = data.ssy - obj.c_var_coefs .* obj.qf_info.qf;
        end

        function obj = update(obj, data, prior, specs)
            move_type = get_move_type(obj.n_ridge, obj.n_quant, prior.n_ridge_max);

            if strcmpi(move_type,'birth')
                % generate birth proposal
                prop = bpprBirthProposal(obj, data, prior, specs);
                if ~isnan(prop.ridge_basis)
                    % update quadratic forms just in case proposal is accepted
                    obj.BtB(1:obj.n_basis_total, prop.basis_idx) = prop.BtP;
                    obj.BtB(prop.basis_idx, 1:obj.n_basis_total) = prop.BtP';
                    obj.BtB(prop.basis_idx, prop.basis_idx) = prop.PtP;
                    obj.Bty(prop.basis_idx) = prop.Pty;

                    % calculate log(MH accpetance probability)
                    prop = prop.get_log_mh(obj, data, prior);

                    if ~isnan(prop.log_mh)
                        if log(rand()) < prop.log_mh
                            obj = obj.acceptBirth(prop, specs.adapt_act_feat);
                        end
                    end
                end
            elseif strcmpi(move_type,'death')
                % generate death proposal 
                prop = bpprDeathProposal(obj, data, prior, specs);

                % calculate log(mh acceptance probability)
                prop = prop.get_log_mh(obj, data, prior);

                if ~isnan(prop.log_mh)
                    if log(rand()) < prop.log_mh
                        obj = obj.acceptDeath(prop, specs.adapt_act_feat);
                    end
                end
            else
                % chage step
                prop = bpprChangeProposal(obj, data, prior, specs);

                if ~isnan(prop.ridge_basis)
                    % calculate log(mh acceptance probability)
                    prop = prop.get_log_mh(obj, data, prior);

                    if ~isnan(prop.log_mh)
                        if log(rand()) < prop.log_mh
                            obj = obj.acceptChange(prop);
                        end
                    end
                end
            end

            if ~strcmpi(obj.phase,'adapt')
                obj = obj.sampleSDresid(data);
                obj = obj.sampleCoefs();

                if strcmpi(prior.prior_coefs,'zs')
                    obj = obj.sampleVarCoefs(data,prior);
                end
            end
        end

    end
end

