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

        function preds = predict(obj, newdata, mcmc_use)
            % BPPR prediction using new inputs (after training).

            % newdata: matrix of predictors with dimension nxp, where n is the number of prediction points and
            %          p is the number of inputs (features). p must match the number of training inputs, and the order of the
            %          columns must also match.
            % mcmc_use: which MCMC samples to use (vector of integers of length m).  Defaults to all MCMC samples.
            arguments
                obj
                newdata
                mcmc_use = nan;
            end

            [n, ~] = size(newdata);

            tmp_obj = obj.data.standardize(newdata);
            newdata_s = tmp_obj.X_st_new;

            if isnan(mcmc_use)
                mcmc_use = 1:obj.specs.n_keep;
            else
                if max(mcmc_use) > (obj.specs.n_keep+1)
                    error('invalid mcmc_use')
                end
            end
            n_use = length(mcmc_use);

            ridge_basis = cell(max(obj.samples.n_ridge),1);
            preds = zeros(n_use,n);

            for i = 1:n_use
                calc_all_bases = false;
                preds(i, :) = obj.samples.coefs(mcmc_use(i), 1);
                if i==1
                    calc_all_bases = true;
                elseif (obj.samples.n_ridge(mcmc_use(i)) ~= obj.samples.n_ridge(mcmc_use(i-1)))
                    calc_all_bases = true;
                end
                if obj.samples.n_ridge(mcmc_use(i)) > 0
                    basis_idx = 1:1;
                    for j = 1:obj.samples.n_ridge(mcmc_use(i))
                        basis_idx = (basis_idx(end)+1):(basis_idx(end)+obj.prior.df_spline);
                        n_act = obj.samples.n_act(mcmc_use(i),j);
                        knots = squeeze(obj.samples.knots(mcmc_use(i),j,:))';
                        if calc_all_bases
                            feat = squeeze(obj.samples.feat(mcmc_use(i),j,1:n_act))';
                            proj_dir = squeeze(obj.samples.proj_dir(mcmc_use(i),j,1:n_act));
                            proj = newdata_s(:,feat) * proj_dir;
                            ridge_basis{j} = get_mns_basis(proj,knots);
                        elseif n_act ~= obj.samples.n_act(mcmc_use(i-1),j) || knots(1) ~= obj.samples.knots(mcmc_use(i-1),j,1)
                            feat = squeeze(obj.samples.feat(mcmc_use(i),j,1:n_act))';
                            proj_dir = squeeze(obj.samples.proj_dir(mcmc_use(i),j,1:n_act));
                            proj = newdata_s(:,feat) * proj_dir;
                            ridge_basis{j} = get_mns_basis(proj,knots);
                        end

                        preds(i, :) = preds(i, :) + (ridge_basis{j} * obj.samples.coefs(mcmc_use(i), basis_idx)')';
                    end
                end
            end

        end

    end
end
