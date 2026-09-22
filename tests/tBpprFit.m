classdef tBpprFit < matlab.unittest.TestCase
    % Integration / smoke tests for the full bppr fit -> predict path.

    methods (TestMethodSetup)
        function seed(~)
            rng(2026);
        end
    end

    methods (Test)
        function fitReturnsModelWithConsistentShapes(tc)
            [X, y, ~, ~] = tBpprFit.makeData(120, 5);
            mod = bppr(X, y, 'n_post', 60, 'n_burn', 60, 'n_thin', 1, ...
                'silent', true);

            tc.verifyClass(mod, 'bpprModel');
            tc.verifyEqual(numel(mod.samples.n_ridge), mod.specs.n_keep);
            tc.verifyEqual(mod.specs.n_keep, 60);

            % n_ridge within [0, n_ridge_max]
            tc.verifyGreaterThanOrEqual(min(mod.samples.n_ridge), 0);
            tc.verifyLessThanOrEqual(max(mod.samples.n_ridge), mod.prior.n_ridge_max);

            % residual variance strictly positive
            tc.verifyTrue(all(mod.samples.s2 > 0));
        end

        function predictShapeAndFinite(tc)
            [X, y, xx, ~] = tBpprFit.makeData(120, 5);
            mod = bppr(X, y, 'n_post', 60, 'n_burn', 60, 'silent', true);
            preds = mod.predict(xx);
            % predict returns [n_keep x nTest]
            tc.verifySize(preds, [mod.specs.n_keep, size(xx,1)]);
            tc.verifyTrue(all(isfinite(preds(:))));
        end

        function predictionTracksTruthLoosely(tc)
            % Generous regression guard: posterior-mean prediction error variance
            % should be much smaller than the response variance.
            [X, y, xx, f] = tBpprFit.makeData(200, 5);
            mod = bppr(X, y, 'n_post', 150, 'n_burn', 150, 'silent', true);
            yhat = mean(mod.predict(xx), 1)';
            errVar = var(yhat - f(xx));
            tc.verifyLessThan(errVar, var(y));
        end

        function flatPriorRuns(tc)
            [X, y, xx, ~] = tBpprFit.makeData(120, 5);
            mod = bppr(X, y, 'prior_coefs', "flat", 'n_post', 40, ...
                'n_burn', 40, 'silent', true);
            tc.verifyClass(mod, 'bpprModel');
            preds = mod.predict(xx);
            tc.verifyTrue(all(isfinite(preds(:))));
        end

        function invalidMcmcUseErrors(tc)
            [X, y, xx, ~] = tBpprFit.makeData(120, 5);
            mod = bppr(X, y, 'n_post', 30, 'n_burn', 30, 'silent', true);
            tc.verifyError(@() mod.predict(xx, 'mcmc_use', mod.specs.n_keep + 5), ?MException);
        end
    end

    methods (Static)
        function [X, y, xx, f] = makeData(n, p)
            % Friedman-style function (reduced-size analog of example.m).
            f = @(x) 10 * sin(pi * x(:,1) .* x(:,2)) + 20 * (x(:,3) - .5).^2 ...
                + 10 * x(:,4) + 5 * x(:,5);
            X = rand(n, p);
            xx = rand(round(n/2), p);
            y = f(X) + randn(n, 1);
        end
    end
end
