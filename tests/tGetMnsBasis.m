classdef tGetMnsBasis < matlab.unittest.TestCase
    % Tests for get_mns_basis.m (natural / modified natural spline basis)

    methods (TestClassSetup)
        function addProjectRootToPath(tc)
            projectRoot = fileparts(fileparts(mfilename('fullpath')));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(projectRoot));
        end
    end

    methods (Test)
        function outputSizeDfFromKnots(tc)
            % df = n_knots - 2
            u = linspace(0, 1, 20)';
            knots = [0 0.25 0.5 0.75 1.0];   % n_knots = 5 -> df = 3
            basis = get_mns_basis(u, knots);
            tc.verifySize(basis, [20, 3]);
        end

        function firstColumnIsRelu(tc)
            u = linspace(-1, 2, 15)';
            knots = [0 0.2 0.4 0.6 0.8 1.0];
            basis = get_mns_basis(u, knots);
            tc.verifyEqual(basis(:,1), relu(u - knots(1)), 'AbsTol', 1e-12);
        end

        function dfEqualsOneBranch(tc)
            % n_knots = 3 -> df = 1 : only the relu column, skips spline block
            u = linspace(0, 1, 10)';
            knots = [0 0.5 1.0];
            basis = get_mns_basis(u, knots);
            tc.verifySize(basis, [10, 1]);
            tc.verifyEqual(basis(:,1), relu(u - knots(1)), 'AbsTol', 1e-12);
        end

        function matchesReferenceImplementation(tc)
            % Independently recompute the algorithm and compare.
            u = linspace(0, 1, 25)';
            knots = [0 0.2 0.4 0.6 0.8 1.0];
            expected = tGetMnsBasis.referenceMns(u, knots);
            tc.verifyEqual(get_mns_basis(u, knots), expected, 'AbsTol', 1e-10);
        end
    end

    methods (Static)
        function basis = referenceMns(u, knots)
            n_knots = numel(knots);
            df = n_knots - 2;
            n = numel(u);
            basis = zeros(n, df);
            basis(:,1) = max(u - knots(1), 0);
            if df > 1
                n_internal_knots = n_knots - 3;
                r = zeros(n, n_knots - 1);
                d = zeros(n, df);
                for k = 2:n_knots
                    r(:,k-1) = max(u - knots(k), 0).^3;
                end
                for k = 1:df
                    d(:,k) = (r(:,k) - r(:,df+1)) ./ (knots(df+2) - knots(k+1));
                end
                for k = 1:n_internal_knots
                    basis(:,k+1) = d(:,k) - d(:,n_internal_knots+1);
                end
            end
        end
    end
end
