classdef bpprFeatProposal
    % Reversible-jump move that adds one input to a ridge function or drops
    % one, keeping the ridge (bpprState.update with specs.p_feat > 0).
    %
    % Add (m -> m+1 active inputs): pick a ridge uniformly among those that
    % may grow (E_add of them) and a new input uniformly among the p - m it
    % does not use; draw u from N(0, s_u^2) truncated to (-1, 1); the new
    % direction is (sqrt(1 - u^2) alpha, u), the new input last, then the
    % inputs are sorted.  Drop (the reverse): pick a ridge uniformly among
    % those with m+1 >= 2 inputs (E_drop of them) and one of its inputs
    % uniformly; u is that input's component and the rest is renormalised.
    % The first knot keeps its relative position in its prior range,
    % knot0 = max_knot0 - v rg, a map whose Jacobian cancels the change in
    % the knot's uniform prior density; the later knots are quantiles of
    % the projection, as always.  The number of spline columns, and so the
    % Zellner-Siow factor, is unchanged.
    %
    % With BayesPPR's prior (number of inputs uniform on 1..n_act_max,
    % inputs a uniform subset, direction uniform on the sphere) the log
    % acceptance ratio of an add is
    %   dll + log C(p,m) - log C(p,m+1) + log|S^(m-1)| - log|S^m|
    %       + log E_add(x) - log E_drop(x') + log(p - m) - log(m + 1)
    %       - log q(u) + (m - 2)/2 log(1 - u^2)
    % (the last term is the Jacobian of (alpha, u) -> alpha' on spheres),
    % and that of a drop is minus the same expression for the reverse add.
    % Only continuous ridges and inputs take part, and m + 1 may reach
    % n_act_max only when n_act_max = p: bppr's birth move treats the inputs
    % of a ridge with n_act_max < p inputs as drawn from its adaptive
    % proposal, not uniformly, so that prior is not the one above.

    properties
        add             % true for an add, false for a drop
        idx_ridge
        n_act
        feat
        proj_dir
        knots
        ridge_basis
        BtB
        Bty
        qf_info
        log_extra       % every term above but dll
        log_mh
        sse
        new_feat        % the input added or dropped
        valid
    end

    methods
        function obj = bpprFeatProposal(state, data, prior, specs, add)
            obj.add = add;
            obj.valid = false;
            p = data.p;
            if any(data.feat_type ~= "cont")
                return
            end
            cap = prior.n_act_max;
            if cap < p, cap = cap - 1; end              % see the class help
            q = state.idx_ridge_quant(:)';
            q = q(state.ridge_type(q) == "cont");
            na = state.n_act(q)';
            grow = q(na < cap);
            shrink = q(na >= 2 & na <= cap);
            su = specs.feat_sd;
            if add
                if isempty(grow), return, end
                k = grow(randi(numel(grow)));
                m = state.n_act(k);
                f0 = state.feat(k, 1:m);
                a0 = state.proj_dir(k, 1:m)';
                free = setdiff(1:p, f0);
                j = free(randi(numel(free)));
                u = trunc_normal(su);
                f1 = [f0, j];
                a1 = [sqrt(1 - u^2) * a0; u];
                [f1, o] = sort(f1);
                a1 = a1(o);
                e_add = numel(grow);
                % ridges that may shrink after the move
                e_drop = numel(shrink) + ~ismember(k, shrink);
            else
                if isempty(shrink), return, end
                k = shrink(randi(numel(shrink)));
                m = state.n_act(k) - 1;                 % inputs after the drop
                f1 = state.feat(k, 1:m+1);
                a1 = state.proj_dir(k, 1:m+1)';
                i = randi(m + 1);
                j = f1(i);
                u = a1(i);
                if abs(u) >= 1, return, end
                f0 = f1([1:i-1, i+1:end]);
                a0 = a1([1:i-1, i+1:end]) / sqrt(1 - u^2);
                if m == 1, a0 = sign(a0); end           % exact +-1 on S^0
                e_drop = numel(shrink);
                % ridges that may grow after the move (k always may)
                e_add = numel(grow) + ~ismember(k, grow);
            end
            % log acceptance of the add x(f0, a0) -> x'(f1, a1), less dll
            lx = lchoose(p, m) - lchoose(p, m + 1) + log_sphere(m - 1) - log_sphere(m) ...
                 + log(e_add) - log(e_drop) + log(p - m) - log(m + 1) ...
                 - log_trunc_normal(u, su) + (m - 2) / 2 * log(1 - u^2);
            if add
                obj.feat = f1; obj.proj_dir = a1; obj.n_act = m + 1; obj.log_extra = lx;
                fold = f0; aold = a0; fnew = f1; anew = a1;
            else
                obj.feat = f0; obj.proj_dir = a0; obj.n_act = m; obj.log_extra = -lx;
                fold = f1; aold = a1; fnew = f0; anew = a0;
            end
            obj.idx_ridge = k;
            obj.new_feat = j;

            % first knot at the same relative position in its prior range
            pold = data.X_st(:, fold) * aold;
            mk0 = quantile(pold, prior.p_dat_max);
            rg0 = (mk0 - min(pold)) ./ prior.prob_relu;
            v = (mk0 - state.knots(k, 1)) / rg0;
            proj = data.X_st(:, fnew) * anew;
            mk1 = quantile(proj, prior.p_dat_max);
            rg1 = (mk1 - min(proj)) ./ prior.prob_relu;
            knot0 = mk1 - v * rg1;
            obj.knots = [knot0, quantile(proj(proj > knot0), prior.knot_quants)];
            if length(unique(obj.knots)) < length(obj.knots)
                return
            end
            obj.ridge_basis = get_mns_basis(proj, obj.knots);
            if prior.center_basis
                obj.ridge_basis = obj.ridge_basis - mean(obj.ridge_basis, 1);
            end
            bi = state.basis_idx{k + 1};
            PtP = obj.ridge_basis' * obj.ridge_basis;
            BtP = state.basis_mat' * obj.ridge_basis;
            obj.BtB = state.BtB(1:state.n_basis_total, 1:state.n_basis_total);
            obj.BtB(bi, :) = BtP';
            obj.BtB(:, bi) = BtP;
            obj.BtB(bi, bi) = PtP;
            obj.Bty = state.Bty(1:state.n_basis_total);
            obj.Bty(bi) = obj.ridge_basis' * data.y;
            obj.valid = true;
        end

        function obj = get_log_mh(obj, state, data, prior)
            obj.qf_info = qf_info(obj.BtB, obj.Bty, prior.intercept);
            obj.log_mh = nan;
            if ~isnan(obj.qf_info.qf) && obj.qf_info.qf < data.ssy
                obj.sse = data.ssy - state.c_var_coefs .* obj.qf_info.qf;
                obj.log_mh = prior.log_lik_sse(obj.sse, data.n) - prior.log_lik_sse(state.sse, data.n) ...
                    + obj.log_extra;
            end
        end
    end
end

function u = trunc_normal(s)
% N(0, s^2) truncated to (-1, 1), by rejection (s is small).
u = 1;
while abs(u) >= 1
    u = s * randn();
end
end

function l = log_trunc_normal(u, s)
z = normcdf(1 / s) - normcdf(-1 / s);
l = -0.5 * (u / s)^2 - log(s) - 0.5 * log(2 * pi) - log(z);
end

function l = log_sphere(k)
% log surface area of the unit sphere S^k in R^(k+1); |S^0| = 2.
l = log(2) + (k + 1) / 2 * log(pi) - gammaln((k + 1) / 2);
end
