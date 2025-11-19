classdef bpprSamples
    % Structure to store data

    properties
        n_ridge
        ridge_type
        n_act
        feat
        proj_dir
        knots
        coefs
        s2
        var_coefs
    end

    methods
        function obj = bpprSamples(prior, specs, state0)
            arguments
                prior
                specs
                state0 = nan
            end
            
            obj.n_ridge = zeros(specs.n_keep,1);
            obj.ridge_type = cell(specs.n_keep,1);
            obj.n_act = zeros(specs.n_keep, prior.n_ridge_max);
            obj.feat = zeros(specs.n_keep, prior.n_ridge_max, prior.n_act_max);
            obj.proj_dir = zeros(specs.n_keep, prior.n_ridge_max, prior.n_act_max);
            obj.knots = zeros(specs.n_keep, prior.n_ridge_max, prior.df_spline+2);
            obj.coefs = zeros(specs.n_keep, prior.df_spline*prior.n_ridge_max + 1);
            if strcmpi(prior.prior_coefs,'zs')
                obj.var_coefs = zeros(specs.n_keep,1);
            elseif strpcmpi(prior.prior_coefs,'flat')
                obj.var_coefs = nan;
            end

            if isa(state0, 'bpprState')
                obj = obj.writeState(state0);
            end
        end

        function obj = writeState(obj, state)
            obj.n_ridge(state.idx) = state.n_ridge;
            obj.n_act(state.idx,:) = state.n_act;
            obj.feat(state.idx,:, :) = state.feat;
            obj.proj_dir(state.idx,:, :) = state.proj_dir;
            obj.knots(state.idx,:,:) = state.knots;
            obj.coefs(state.idx, :) = state.coefs;
            obj.s2(state.idx) = state.s2;
            if ~isnan(state.var_coefs)
                obj.var_coefs(state.idx) = state.var_coefs;
            end
        end

    end
end
