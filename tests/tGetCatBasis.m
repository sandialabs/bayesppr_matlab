classdef tGetCatBasis < matlab.unittest.TestCase
    % Tests for get_cat_basis.m

    methods (Test)
        function singleColumnUnchanged(tc)
            Xj = [0; 1; 1; 0];
            tc.verifyEqual(get_cat_basis(Xj), Xj, 'AbsTol', 1e-12);
        end

        function multiColumnInclusionExclusion(tc)
            % basis = 1 - prod(1 - Xj, 2)  ("any feature active")
            Xj = [0 0; 1 0; 0 1; 1 1];
            expected = [0; 1; 1; 1];
            tc.verifyEqual(get_cat_basis(Xj), expected, 'AbsTol', 1e-12);
        end

        function threeColumns(tc)
            Xj = [0 0 0; 1 0 0; 0 1 1; 1 1 1];
            expected = 1 - prod(1 - Xj, 2);
            tc.verifyEqual(get_cat_basis(Xj), expected, 'AbsTol', 1e-12);
            tc.verifyEqual(expected, [0; 1; 1; 1], 'AbsTol', 1e-12);
        end

        function outputIsColumn(tc)
            Xj = [0 1; 1 1; 0 0];
            out = get_cat_basis(Xj);
            tc.verifySize(out, [3 1]);
        end
    end
end
