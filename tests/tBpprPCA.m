classdef tBpprPCA < matlab.unittest.TestCase
    % Integration tests for bpprPCA / bpprPCAsetup (multivariate response).

    methods (TestMethodSetup)
        function seed(~)
            rng(3030);
        end
    end

    methods (Test)
        function pcaSetupReconstructsInput(tc)
            y = tBpprPCA.makeResponse(60);
            setup = bpprPCAsetup(y, true, false);
            % basis * newy reconstructs the centered/scaled response (transposed)
            recon = setup.basis * setup.newy;
            tc.verifyEqual(recon, setup.y_scale', 'AbsTol', 1e-8);
        end

        function eigenvaluesSortedDescending(tc)
            y = tBpprPCA.makeResponse(60);
            setup = bpprPCAsetup(y, true, false);
            tc.verifyTrue(all(diff(setup.evals) <= 1e-9));
            tc.verifyTrue(all(setup.evals >= -1e-9));
        end

        function fitReturnsBpprBasis(tc)
            [x, y, ~] = tBpprPCA.makeData(120, 4);
            mod = bpprPCA(x, y, NaN, 99.9, 1, true, false, ...
                'n_post', 40, 'n_burn', 40, 'silent', true);
            tc.verifyClass(mod, 'bpprBasis');
            tc.verifyGreaterThanOrEqual(mod.nbasis, 1);
        end

        function predictShapeAndFinite(tc)
            [x, y, xx] = tBpprPCA.makeData(120, 4);
            q = size(y, 2);
            mod = bpprPCA(x, y, NaN, 99.9, 1, true, false, ...
                'n_post', 40, 'n_burn', 40, 'silent', true);
            preds = mod.predict(xx);
            % [nTest x q x n_keep]
            tc.verifyEqual(size(preds, 1), size(xx, 1));
            tc.verifyEqual(size(preds, 2), q);
            tc.verifyEqual(size(preds, 3), mod.bm_list{1}.specs.n_keep);
            tc.verifyTrue(all(isfinite(preds(:))));
        end
    end

    methods (Static)
        function y = makeResponse(n)
            % n x q multivariate response (reduced-size analog of examplePCA.m).
            f = @(x) 10 .* sin(pi .* linspace(0,1,20) .* x(:,1)) ...
                + 20 .* (x(:,2) - .5).^2 + 10 .* x(:,3) + 5 .* x(:,4);
            x = rand(n, 4) - 0.5;
            y = f(x);
        end

        function [x, y, xx] = makeData(n, p)
            f = @(x) 10 .* sin(pi .* linspace(0,1,20) .* x(:,1)) ...
                + 20 .* (x(:,2) - .5).^2 + 10 .* x(:,3) + 5 .* x(:,4);
            x = rand(n, p) - 0.5;
            xx = rand(round(n/2), p) - 0.5;
            y = f(x);
        end
    end
end
