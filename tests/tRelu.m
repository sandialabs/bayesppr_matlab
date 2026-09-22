classdef tRelu < matlab.unittest.TestCase
    % Tests for relu.m

    methods (TestClassSetup)
        function addProjectRootToPath(tc)
            % Source functions live in the project root (parent of tests/).
            projectRoot = fileparts(fileparts(mfilename('fullpath')));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(projectRoot));
        end
    end

    methods (Test)
        function positiveUnchanged(tc)
            x = [0.5; 1; 100];
            tc.verifyEqual(relu(x), x, 'AbsTol', 1e-12);
        end

        function negativeZeroed(tc)
            x = [-0.5; -1; -100];
            tc.verifyEqual(relu(x), zeros(3,1), 'AbsTol', 1e-12);
        end

        function zeroIsZero(tc)
            tc.verifyEqual(relu(0), 0, 'AbsTol', 1e-12);
        end

        function matchesMaxElementwise(tc)
            x = [-3 -1 0; 0.5 2 -7];
            tc.verifyEqual(relu(x), max(x, 0), 'AbsTol', 1e-12);
        end

        function preservesShape(tc)
            x = randn(4, 5);
            tc.verifySize(relu(x), size(x));
        end
    end
end
