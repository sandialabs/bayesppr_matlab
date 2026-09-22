classdef tGetMoveType < matlab.unittest.TestCase
    % Tests for get_move_type.m

    methods (TestMethodSetup)
        function seed(~)
            rng(20260921);
        end
    end

    methods (Test)
        function zeroRidgeIsBirth(tc)
            tc.verifyEqual(get_move_type(0, 0, 10), "birth");
            tc.verifyEqual(get_move_type(0, 5, 10), "birth");
        end

        function fullNoQuantIsDeath(tc)
            tc.verifyEqual(get_move_type(10, 0, 10), "death");
        end

        function fullWithQuantDeathOrChange(tc)
            allowed = ["death", "change"];
            for i = 1:200
                mt = get_move_type(10, 3, 10);
                tc.verifyTrue(ismember(mt, allowed), char(mt));
            end
        end

        function midNoQuantBirthOrDeath(tc)
            allowed = ["birth", "death"];
            for i = 1:200
                mt = get_move_type(5, 0, 10);
                tc.verifyTrue(ismember(mt, allowed), char(mt));
            end
        end

        function midWithQuantAnyMove(tc)
            allowed = ["birth", "death", "change"];
            for i = 1:200
                mt = get_move_type(5, 2, 10);
                tc.verifyTrue(ismember(mt, allowed), char(mt));
            end
        end
    end
end
