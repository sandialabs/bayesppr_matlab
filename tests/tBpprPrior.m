classdef tBpprPrior < matlab.unittest.TestCase
    % Tests for bpprPrior.m (constructor + calibrate)

    methods (Test)
        function knotQuantsFromDfSpline(tc)
            prior = tBpprPrior.makePrior(4);
            tc.verifyEqual(prior.knot_quants, linspace(0, 1, 5), 'AbsTol', 1e-12);
        end

        function zsDefaultsCalibrated(tc)
            data = tBpprPrior.makeData(100, 4);
            prior = tBpprPrior.makePrior(4);
            prior = prior.calibrate(data);

            tc.verifyEqual(prior.shape_var_coefs, 0.5);
            tc.verifyEqual(prior.scale_var_coefs, data.n / 2);
        end

        function nDatMinDefault(tc)
            data = tBpprPrior.makeData(300, 4);   % 0.1*n = 30 -> min(20,30)=20
            prior = tBpprPrior.makePrior(4);
            prior = prior.calibrate(data);
            tc.verifyEqual(prior.n_dat_min, 20);
            tc.verifyEqual(prior.p_dat_max, 1 - 20/300, 'RelTol', 1e-12);
        end

        function nActMaxFormula(tc)
            % 4 features, none categorical: min(3,4-0)+min(3,ceil(0/2)) = 3+0 = 3
            data = tBpprPrior.makeData(100, 4);
            prior = tBpprPrior.makePrior(4);
            prior = prior.calibrate(data);
            n_cat = sum(data.feat_type == "cat");
            expected = min(3, data.p - n_cat) + min(3, ceil(n_cat/2));
            tc.verifyEqual(prior.n_act_max, expected);
        end

        function nRidgeMaxFormula(tc)
            data = tBpprPrior.makeData(100, 4);
            prior = tBpprPrior.makePrior(4);
            prior = prior.calibrate(data);
            tc.verifyEqual(prior.n_ridge_max, min(150, floor(data.n/4) - 2));
        end

        function projDirMnSizes(tc)
            data = tBpprPrior.makeData(100, 4);
            prior = tBpprPrior.makePrior(4);
            prior = prior.calibrate(data);
            for a = 1:prior.n_act_max
                tc.verifyEqual(prior.proj_dir_mn{a}, repelem(1/sqrt(a), a), ...
                    'AbsTol', 1e-12);
            end
        end

        function smallNDatMinWarns(tc)
            % df_spline large vs n forces n_dat_min <= df_spline warning branch.
            % The warning is message-only (no identifier), so capture it via lastwarn.
            data = tBpprPrior.makeData(60, 20);
            prior = bpprPrior(10, nan, nan, 20, 2/3, "zs", nan, nan, nan);
            lastwarn('');
            wstate = warning('off', 'all');
            cleanup = onCleanup(@() warning(wstate));
            prior = prior.calibrate(data);
            [msg, ~] = lastwarn();
            tc.verifyNotEmpty(msg);
            tc.verifyEqual(prior.n_dat_min, 21);   % df_spline + 1
        end

        function nRidgeMaxNonPositiveErrors(tc)
            % df_spline too large -> floor(n/df_spline)-2 <= 0.
            % The error is message-only, so match on any MException.
            data = tBpprPrior.makeData(30, 25);
            prior = bpprPrior(10, nan, nan, 25, 2/3, "zs", nan, nan, nan);
            tc.verifyError(@() prior.calibrate(data), ?MException);
        end
    end

    methods (Static)
        function prior = makePrior(df_spline)
            prior = bpprPrior(10, nan, nan, df_spline, 2/3, "zs", nan, nan, nan);
        end

        function data = makeData(n, df_spline)
            rng(99);
            X = randn(n, 4);
            y = randn(n, 1);
            prior = tBpprPrior.makePrior(df_spline);
            data = bpprData(X, y);
            data = data.summarize(prior);
        end
    end
end
