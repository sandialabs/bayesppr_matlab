classdef tBpprGibbs < matlab.unittest.TestCase
    % Tests for the Gibbs-block options: intercept, center_basis, the s2
    % prior, and the bpprGibbs stepping class.

    methods (TestClassSetup)
        function addProjectRootToPath(tc)
            projectRoot = fileparts(fileparts(mfilename('fullpath')));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(projectRoot));
        end
    end

    methods (TestMethodSetup)
        function seed(~)
            rng(2026);
        end
    end

    methods (Test)
        function logLikSseDefaultMatchesOriginal(tc)
            % The default prior must give the original acceptance ratio.
            prior = bpprPrior(10, nan, nan, 4, 2/3, "zs", nan, nan, nan);
            n = 50; a = 3.7; b = 2.2;
            tc.verifyEqual(prior.log_lik_sse(a, n) - prior.log_lik_sse(b, n), ...
                -n/2 * (log(a) - log(b)), 'AbsTol', 1e-12);
        end

        function logLikSseInverseGammaAndFixed(tc)
            prior = bpprPrior(10, nan, nan, 4, 2/3, "zs", nan, nan, nan);
            prior.shape_s2 = 2; prior.scale_s2 = 0.5;
            tc.verifyEqual(prior.log_lik_sse(3, 10), -(2 + 5) * log(0.5 + 1.5), 'AbsTol', 1e-12);
            prior.s2_fixed = 0.25;
            tc.verifyEqual(prior.log_lik_sse(3, 10), -3 / 0.5, 'AbsTol', 1e-12);
        end

        function defaultFitUnchangedShapes(tc)
            % Explicit defaults behave like no options at all.
            [X, y] = tBpprGibbs.makeData(100);
            mod = bppr(X, y, 'n_post', 40, 'n_burn', 40, 'silent', true, ...
                'intercept', true, 'center_basis', false);
            tc.verifyTrue(all(isfinite(mod.predict(X), 'all')));
            tc.verifyTrue(all(mod.samples.s2 > 0));
        end

        function centeredNoInterceptIsMeanZero(tc)
            % intercept = false, center_basis = true: every posterior draw of
            % the regression function averages to zero over the training X.
            [X, y] = tBpprGibbs.makeData(100);
            y = y - mean(y);
            mod = bppr(X, y, 'n_post', 60, 'n_burn', 100, 'silent', true, ...
                'intercept', false, 'center_basis', true);
            P = mod.predict(X);
            tc.verifyLessThan(max(abs(mean(P, 2))), 1e-10);
            tc.verifyGreaterThan(max(mod.samples.n_ridge), 0);
            % and it still fits: prediction error well below var(y)
            tc.verifyLessThan(mean((mean(P, 1)' - y).^2), 0.5 * var(y));
        end

        function fixedS2StaysFixed(tc)
            [X, y] = tBpprGibbs.makeData(80);
            mod = bppr(X, y, 'n_post', 30, 'n_burn', 30, 'silent', true, ...
                's2_fixed', 0.7);
            tc.verifyEqual(mod.samples.s2, 0.7 * ones(30, 1));
        end

        function inverseGammaPriorRuns(tc)
            [X, y] = tBpprGibbs.makeData(80);
            mod = bppr(X, y, 'n_post', 30, 'n_burn', 30, 'silent', true, ...
                'shape_s2', 2, 'scale_s2', 0.1);
            tc.verifyTrue(all(mod.samples.s2 > 0));
        end

        function gibbsFittedMatchesPredict(tc)
            % Each saved state's fitted() must equal predict() of that draw at
            % the training inputs, with and without intercept/centering.
            [X, y] = tBpprGibbs.makeData(80);
            for cfg = {{true, false}, {false, true}}
                g = bpprGibbs(X, y - mean(y), 10, 'n_ridge_mean', 3, ...
                    'intercept', cfg{1}{1}, 'center_basis', cfg{1}{2});
                mu = zeros(10, numel(y));
                for k = 1:10
                    % response moves between visits, as in a Gibbs sampler
                    g.setResponse(y - mean(y) + 0.1 * randn(size(y)));
                    g.step(5);
                    mu(k, :) = g.fitted()';
                    g.save();
                end
                P = g.model().predict(X);
                tc.verifyEqual(P, mu, 'AbsTol', 1e-8);
            end
        end

        function setResponseMatchesFreshState(tc)
            % After setResponse, the cached sse equals that of a state built
            % directly on the new response with the same basis.
            [X, y] = tBpprGibbs.makeData(80);
            g = bpprGibbs(X, y, 1, 'n_ridge_mean', 3);
            g.step(20);
            y2 = y + randn(size(y));
            g.setResponse(y2);
            nb = g.state.n_basis_total;
            B = g.state.basis_mat;
            bhat = (B' * B) \ (B' * y2);
            sse = y2' * y2 - g.state.c_var_coefs * (y2' * B * bhat);
            tc.verifyEqual(g.state.sse, sse, 'RelTol', 1e-8);
            tc.verifyEqual(g.state.Bty(1:nb), B' * y2, 'RelTol', 1e-10);
        end

        function gibbsRecoversLatentRegression(tc)
            % A toy data-augmentation sampler: z_i ~ N(f(x_i), s2), y_i ~
            % N(z_i, 0.1^2), with z drawn exactly given (f, s2).  The
            % BayesPPR block should track f.
            [X, ~, f] = tBpprGibbs.makeData(150);
            ftrue = f(X) - mean(f(X));
            y = ftrue + 0.3 * randn(size(ftrue)) + 0.1 * randn(size(ftrue));
            g = bpprGibbs(X, y, 50, 'n_ridge_mean', 3, 'intercept', false, ...
                'center_basis', true, 'shape_s2', 2, 'scale_s2', 0.05);
            z = y;
            for it = 1:300
                g.setResponse(z);
                g.step(3);
                mu = g.fitted(); s2 = g.s2();
                v = 1 / (1 / s2 + 1 / 0.01);
                z = v * (mu / s2 + y / 0.01) + sqrt(v) * randn(size(y));
                if it > 250, g.save(); end
            end
            muhat = mean(g.model().predict(X), 1)';
            tc.verifyLessThan(mean((muhat - ftrue).^2), 0.25 * var(ftrue));
        end

        function flatPriorRejected(tc)
            [X, y] = tBpprGibbs.makeData(50);
            tc.verifyError(@() bpprGibbs(X, y, 5, 'prior_coefs', "flat"), ...
                'bpprGibbs:flat');
        end
    end

    methods (Static)
        function [X, y, f] = makeData(n)
            f = @(x) 10 * sin(pi * x(:,1) .* x(:,2)) + 20 * (x(:,3) - .5).^2;
            X = rand(n, 3);
            y = f(X) + randn(n, 1);
        end
    end
end
