classdef tLchoose < matlab.unittest.TestCase
    % Tests for lchoose.m
    % lchoose(n,k) = -betaln(1+n-k, 1+k) - log(n+1)
    % Since B(n-k+1, k+1) = 1 / ((n+1) * nchoosek(n,k)),
    %   -betaln(...) = log((n+1) * nchoosek(n,k)),
    % so lchoose(n,k) = log(nchoosek(n,k)).

    methods (TestClassSetup)
        function addProjectRootToPath(tc)
            projectRoot = fileparts(fileparts(mfilename('fullpath')));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(projectRoot));
        end
    end

    methods (Test)
        function matchesNchoosek(tc)
            cases = [5 2; 10 0; 10 10; 8 3; 20 7];
            for i = 1:size(cases,1)
                n = cases(i,1);
                k = cases(i,2);
                expected = log(nchoosek(n,k));
                tc.verifyEqual(lchoose(n,k), expected, 'AbsTol', 1e-10, ...
                    sprintf('n=%d k=%d', n, k));
            end
        end

        function matchesBetalnForm(tc)
            n = 12; k = 5;
            expected = -betaln(1+n-k, 1+k) - log(n+1);
            tc.verifyEqual(lchoose(n,k), expected, 'RelTol', 1e-12);
        end

        function symmetryInK(tc)
            % nchoosek(n,k) == nchoosek(n,n-k)
            n = 15;
            tc.verifyEqual(lchoose(n,4), lchoose(n,11), 'RelTol', 1e-10);
        end
    end
end
