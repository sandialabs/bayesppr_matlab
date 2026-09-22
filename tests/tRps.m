classdef tRps < matlab.unittest.TestCase
    % Tests for rps.m (draw from power-spherical distribution)

    methods (TestClassSetup)
        function addProjectRootToPath(tc)
            projectRoot = fileparts(fileparts(mfilename('fullpath')));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(projectRoot));
        end
    end

    methods (TestMethodSetup)
        function seed(~)
            rng(4242);
        end
    end

    methods (Test)
        function returnsUnitColumnVector(tc)
            for d = 2:6
                mu = ones(d,1) / sqrt(d);
                theta = rps(mu, 0.0);
                tc.verifySize(theta, [d 1]);
                tc.verifyEqual(norm(theta), 1, 'AbsTol', 1e-9, ...
                    sprintf('d=%d', d));
            end
        end

        function unitNormAcrossKappa(tc)
            mu = [1; 0; 0];
            for kappa = [0, 1, 10, 1000]
                theta = rps(mu, kappa);
                tc.verifyEqual(norm(theta), 1, 'AbsTol', 1e-9, ...
                    sprintf('kappa=%g', kappa));
            end
        end

        function highConcentrationNearMu(tc)
            % With very large kappa the draw should concentrate near mu.
            mu = [1; 0; 0];
            acc = zeros(3,1);
            for i = 1:50
                acc = acc + rps(mu, 1e5);
            end
            meanDir = acc / 50;
            meanDir = meanDir / norm(meanDir);
            tc.verifyGreaterThan(mu' * meanDir, 0.9);
        end
    end
end
