classdef tBpprSpecs < matlab.unittest.TestCase
    % Tests for bpprSpecs.m (constructor + calibrate)

    methods (TestClassSetup)
        function addProjectRootToPath(tc)
            projectRoot = fileparts(fileparts(mfilename('fullpath')));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(projectRoot));
        end
    end

    methods (Test)
        function thinningMath(tc)
            % n_post = 100, n_thin = 3 -> n_post reduced to 99, n_keep = 33
            s = bpprSpecs(100, 50, 10, 3, nan, nan, true, nan);
            tc.verifyEqual(s.n_post, 99);
            tc.verifyEqual(s.n_keep, 33);
            tc.verifyEqual(s.n_pre, 60);          % n_adapt + n_burn
            tc.verifyEqual(s.n_draws, 60 + 99);   % n_pre + n_post
        end

        function noThinning(tc)
            s = bpprSpecs(200, 100, 0, 1, nan, nan, true, nan);
            tc.verifyEqual(s.n_post, 200);
            tc.verifyEqual(s.n_keep, 200);
            tc.verifyEqual(s.n_pre, 100);
            tc.verifyEqual(s.n_draws, 300);
        end

        function thinGreaterThanPostErrors(tc)
            tc.verifyError(@() bpprSpecs(10, 5, 0, 20, nan, nan, true, nan), ?MException);
        end

        function defaultProjDirPrecision(tc)
            s = bpprSpecs(100, 50, 0, 1, nan, nan, true, nan);
            tc.verifyEqual(s.proj_dir_prop_prec, 1000.0);
        end

        function scaleProjDirOutOfRangeErrors(tc)
            tc.verifyError(@() bpprSpecs(100, 50, 0, 1, nan, nan, true, 1.5), ?MException);
            tc.verifyError(@() bpprSpecs(100, 50, 0, 1, nan, nan, true, 0), ?MException);
        end

        function scaleProjDirValidComputesPrecision(tc)
            scale = 0.002;
            s = bpprSpecs(100, 50, 0, 1, nan, nan, true, scale);
            inv = 1/scale;
            expected = (inv - 1) + sqrt(inv * (inv - 1));
            tc.verifyEqual(s.proj_dir_prop_prec, expected, 'RelTol', 1e-10);
        end

        function calibrateZeroesConstantFeatureWeights(tc)
            [data, prior] = tBpprSpecs.makeDataPrior();
            s = bpprSpecs(100, 50, 0, 1, nan, nan, true, nan);
            s = s.calibrate(data, prior);
            tc.verifySize(s.w_feat, [data.p 1]);
            % column 1 is constant -> weight zeroed
            tc.verifyEqual(s.w_feat(1), 0.0);
            tc.verifyEqual(s.w_feat(2), 1.0);
            tc.verifyEqual(s.w_n_act, ones(prior.n_act_max, 1));
        end

        function calibrateFlatSetsAdaptAndZeroBurn(tc)
            [data, ~] = tBpprSpecs.makeDataPrior();
            priorFlat = bpprPrior(10, nan, nan, 4, 2/3, "flat", nan, nan, nan);
            priorFlat = priorFlat.calibrate(data);
            s = bpprSpecs(100, 50, 10, 1, nan, nan, true, nan);
            n_pre_before = s.n_pre;
            s = s.calibrate(data, priorFlat);
            tc.verifyEqual(s.n_adapt, n_pre_before);
            tc.verifyEqual(s.n_burn, 0);
        end
    end

    methods (Static)
        function [data, prior] = makeDataPrior()
            rng(123);
            n = 60;
            X = [ones(n,1), randn(n,1), randn(n,1), randn(n,1)];  % col 1 constant
            y = randn(n,1);
            prior = bpprPrior(10, nan, nan, 4, 2/3, "zs", nan, nan, nan);
            data = bpprData(X, y);
            data = data.summarize(prior);
            prior = prior.calibrate(data);
        end
    end
end
