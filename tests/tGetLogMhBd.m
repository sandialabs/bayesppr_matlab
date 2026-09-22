classdef tGetLogMhBd < matlab.unittest.TestCase
    % Tests for get_log_mh_bd.m

    methods (TestClassSetup)
        function addProjectRootToPath(tc)
            projectRoot = fileparts(fileparts(mfilename('fullpath')));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(projectRoot));
        end
    end

    methods (Test)
        function zeroRidgeProp(tc)
            tc.verifyEqual(get_log_mh_bd(0, 0, 10), 0);
            tc.verifyEqual(get_log_mh_bd(0, 4, 10), 0);
        end

        function fullNoQuant(tc)
            tc.verifyEqual(get_log_mh_bd(10, 0, 10), 0);
        end

        function fullWithQuant(tc)
            tc.verifyEqual(get_log_mh_bd(10, 3, 10), log(2), 'AbsTol', 1e-12);
        end

        function midNoQuant(tc)
            tc.verifyEqual(get_log_mh_bd(5, 0, 10), log(2), 'AbsTol', 1e-12);
        end

        function midWithQuant(tc)
            tc.verifyEqual(get_log_mh_bd(5, 2, 10), log(3), 'AbsTol', 1e-12);
        end
    end
end
