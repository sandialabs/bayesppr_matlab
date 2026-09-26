classdef tDwallenius < matlab.unittest.TestCase
    % Tests for dwallenius.m
    % Multivariate Wallenius' noncentral hypergeometric density with some
    % variables fixed.

    methods (TestClassSetup)
        function addProjectRootToPath(tc)
            projectRoot = fileparts(fileparts(mfilename('fullpath')));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(projectRoot));
        end
    end

    methods (Test)
        function allFeaturesSelectedReturnsOne(tc)
            wfeat_norm = [0.25 0.25 0.25 0.25];
            feat = 1:4;   % length(feat) == length(wfeat_norm)
            tc.verifyEqual(dwallenius(wfeat_norm, feat), 1);
        end

        function subsetMatchesReference(tc)
            wfeat_norm = [0.1 0.2 0.3 0.4];
            feat = [1 3];
            expected = tDwallenius.referenceDwallenius(wfeat_norm, feat);
            tc.verifyEqual(dwallenius(wfeat_norm, feat), expected, 'RelTol', 1e-10);
        end

        function singleFeatureSubset(tc)
            wfeat_norm = [0.2 0.3 0.5];
            feat = 2;
            % j = 1: ss = 1 + (-1)^1 / (sum(logMH) + 1), no inner loop
            expected = tDwallenius.referenceDwallenius(wfeat_norm, feat);
            tc.verifyEqual(dwallenius(wfeat_norm, feat), expected, 'RelTol', 1e-10);
        end

        function largerSubset(tc)
            wfeat_norm = [0.05 0.1 0.15 0.25 0.2 0.25];
            feat = [2 4 5];
            expected = tDwallenius.referenceDwallenius(wfeat_norm, feat);
            tc.verifyEqual(dwallenius(wfeat_norm, feat), expected, 'RelTol', 1e-10);
        end
    end

    methods (Static)
        function ss = referenceDwallenius(wfeat_norm, feat)
            if numel(feat) == numel(wfeat_norm)
                ss = 1;
                return;
            end
            logMH = wfeat_norm(feat) ./ (1 - sum(wfeat_norm(feat)));
            logMH = logMH(:);
            j = numel(logMH);
            ss = 1 + (-1)^j / (sum(logMH) + 1);
            for i = 1:(j-1)
                idx = nchoosek(1:j, i);
                temp = reshape(logMH(idx), size(idx));
                ss = ss + (-1)^i * sum(1 ./ (sum(temp, 2) + 1));
            end
        end
    end
end
