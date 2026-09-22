classdef tBpprData < matlab.unittest.TestCase
    % Tests for bpprData.m (summarize / standardize)

    methods (Test)
        function featureTypeClassification(tc)
            n = 40;
            rng(11);
            % col 1: constant  -> ""
            % col 2: binary    -> "cat"
            % col 3: 3 unique  -> "disc" (<= df_spline)
            % col 4: continuous-> "cont"
            X = [ones(n,1), ...
                 repmat([0;1], n/2, 1), ...
                 repmat([1;2;3;3], n/4, 1), ...
                 randn(n,1)];
            y = randn(n,1);
            prior = tBpprData.makePrior(4);

            data = bpprData(X, y);
            data = data.summarize(prior);

            tc.verifyEqual(data.feat_type(1), "");
            tc.verifyEqual(data.feat_type(2), "cat");
            tc.verifyEqual(data.feat_type(3), "disc");
            tc.verifyEqual(data.feat_type(4), "cont");
        end

        function nPSsyComputed(tc)
            X = [randn(25,1), randn(25,1) + 3, randn(25,1)];
            y = (1:25)';
            data = bpprData(X, y);
            data = data.summarize(tBpprData.makePrior(4));
            tc.verifyEqual(data.n, 25);
            tc.verifyEqual(data.p, 3);
            tc.verifyEqual(data.ssy, y' * y, 'RelTol', 1e-12);
        end

        function constantAndBinaryNotStandardized(tc)
            n = 20;
            X = [ones(n,1), repmat([0;1], n/2, 1), randn(n,1) * 5 + 10];
            y = randn(n,1);
            data = bpprData(X, y);
            data = data.summarize(tBpprData.makePrior(4));
            % constant + binary keep mn=0, sd=1
            tc.verifyEqual(data.mn_X(1), 0);
            tc.verifyEqual(data.sd_X(1), 1);
            tc.verifyEqual(data.mn_X(2), 0);
            tc.verifyEqual(data.sd_X(2), 1);
            % continuous column is standardized with mean/std
            tc.verifyEqual(data.mn_X(3), mean(X(:,3)), 'RelTol', 1e-12);
            tc.verifyEqual(data.sd_X(3), std(X(:,3)), 'RelTol', 1e-12);
        end

        function standardizeMatchesZScore(tc)
            n = 30;
            X = [randn(n,1)*2 + 1, randn(n,1)*3 - 4, randn(n,1)];
            y = randn(n,1);
            data = bpprData(X, y);
            data = data.summarize(tBpprData.makePrior(4));

            expected = (X - data.mn_X') ./ data.sd_X';
            tc.verifyEqual(data.X_st, expected, 'RelTol', 1e-10);
        end

        function standardizeNewData(tc)
            n = 30;
            X = [randn(n,1)*2 + 1, randn(n,1)*3 - 4, randn(n,1)];
            y = randn(n,1);
            data = bpprData(X, y);
            data = data.summarize(tBpprData.makePrior(4));

            Xnew = randn(7, 3);
            tmp = data.standardize(Xnew);
            expected = (Xnew - data.mn_X') ./ data.sd_X';
            tc.verifyEqual(tmp.X_st_new, expected, 'RelTol', 1e-10);
        end
    end

    methods (Static)
        function prior = makePrior(df_spline)
            prior = bpprPrior(10, nan, nan, df_spline, 2/3, "zs", nan, nan, nan);
        end
    end
end
