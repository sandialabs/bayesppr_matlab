classdef bpprModel < handle
    % The model structure, including the current RJMCMC state and previous saved states; with methods for saving the
    % state, plotting MCMC traces, and predicting

    properties
        data
        prior
        specs
        samples
    end

    methods
        function obj = bpprModel(data, prior, specs, samples)
            obj.data = data;
            obj.prior = prior;
            obj.specs = specs;
            obj.samples = samples;
        end

        function plot(obj)
            % Trace plots and predictions/residuals

            % * top left - trace plot of number of basis functions (excluding burn-in and thinning)
            % * top right - trace plot of residual variance
            % * bottom left - training data against predictions
            % * bottom right - histogram of residuals (posterior mean) with assumed Gaussian overlaid.

            figure()
            subplot(2,2,1)
            plot(obj.samples.n_ridge)
            ylabel('number of ridge functions')
            xlabel('MCMC iteration (post-burn)')

            subplot(2,2,2)
            plot(obj.samples.s2)
            ylabel('error variance')
            xlabel('MCMC iteration (post-burn)')

            subplot(2,2,3)
            yhat = mean(obj.predict(obj.data.X),1);
            scatter(obj.data.y, yhat)
            refline(1,0)
            xlabel('observed')
            ylabel('posterior prediction')

            subplot(2,2,4)
            histfit(obj.data.y(:)-yhat(:))
            xlabel('residuals')
            ylabel('density')
        end

        function preds = predict(obj, newdata, options)
            % BPPR prediction using new inputs (after training).

            % newdata: matrix of predictors with dimension nxp, where n is the number of prediction points and
            %          p is the number of inputs (features). p must match the number of training inputs, and the order of the
            %          columns must also match.
            % mcmc_use: which MCMC samples to use (vector of integers of length m).  Defaults to all MCMC samples.
            arguments
                obj
                newdata
                options.mcmc_use = NaN
            end
            mcmc_use = options.mcmc_use;

            [n, ~] = size(newdata);

            tmp_obj = obj.data.standardize(newdata);
            newdata_s = tmp_obj.X_st_new;

            if isscalar(mcmc_use) && isnan(mcmc_use)
                mcmc_use = 1:obj.specs.n_keep;
            else
                if max(mcmc_use) > obj.specs.n_keep
                    error('invalid mcmc_use')
                end
            end
            n_use = length(mcmc_use);

            % With center_basis, each ridge basis column had its training mean
            % subtracted during fitting, so the same means (recomputed from
            % the stored training inputs) are subtracted here.
            center = obj.prior.center_basis;
            n_int = double(obj.prior.intercept);
            X_train = obj.data.X_st;

            n_ridge_max = max([obj.samples.n_ridge(mcmc_use); 0]);
            ridge_basis = cell(n_ridge_max, 1);
            ridge_key = cell(n_ridge_max, 1);
            preds = zeros(n_use,n);

            for i = 1:n_use
                s = mcmc_use(i);
                if n_int > 0
                    preds(i, :) = obj.samples.coefs(s, 1);
                end
                n_basis = n_int;
                for j = 1:obj.samples.n_ridge(s)
                    ridge_type = obj.samples.ridge_type{s}(j);
                    n_act = obj.samples.n_act(s,j);
                    feat = squeeze(obj.samples.feat(s,j,1:n_act))';
                    proj_dir = reshape(obj.samples.proj_dir(s,j,1:n_act), [], 1);
                    knots = reshape(obj.samples.knots(s,j,:), 1, []);
                    if ridge_type == "cont"
                        nb_j = obj.prior.df_spline;
                    else
                        nb_j = 1;
                    end
                    basis_idx = (n_basis+1):(n_basis+nb_j);
                    n_basis = n_basis + nb_j;

                    % Recompute the basis only when this ridge function differs
                    % from the one cached at position j.
                    key = [double(ridge_type == "cont") + 2*double(ridge_type == "cat"), ...
                        n_act, feat, proj_dir', knots];
                    if ~isequaln(key, ridge_key{j})   % NaN knots/directions for "cat"
                        ridge_basis{j} = bpprModel.ridgeBasis(newdata_s, ridge_type, feat, proj_dir, knots);
                        if center
                            ridge_basis{j} = ridge_basis{j} - ...
                                mean(bpprModel.ridgeBasis(X_train, ridge_type, feat, proj_dir, knots), 1);
                        end
                        ridge_key{j} = key;
                    end

                    preds(i, :) = preds(i, :) + (ridge_basis{j} * obj.samples.coefs(s, basis_idx)')';
                end
            end

        end

    end

    methods (Static)
        function B = ridgeBasis(Xs, ridge_type, feat, proj_dir, knots)
            % Basis columns of one ridge function at standardized inputs Xs.
            if ridge_type == "cont"
                B = get_mns_basis(Xs(:, feat) * proj_dir, knots);
            elseif ridge_type == "cat"
                B = get_cat_basis(Xs(:, feat));
            else
                B = Xs(:, feat) * proj_dir;
            end
        end
    end
end
