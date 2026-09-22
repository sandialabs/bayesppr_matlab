classdef tQfInfo < matlab.unittest.TestCase
    % Tests for qf_info.m

    methods (Test)
        function wellConditionedQuadraticForm(tc)
            rng(7);
            B = randn(30, 4);
            Bty = B' * randn(30, 1);
            BtB = B' * B;
            info = qf_info(BtB, Bty);

            tc.verifyTrue(info.fullrank);
            tc.verifyEqual(info.dim, 4);
            % qf = Bty' * (BtB \ Bty)
            tc.verifyEqual(info.qf, Bty' * (BtB \ Bty), 'RelTol', 1e-8);
            % least-squares estimate
            tc.verifyEqual(info.ls_est, BtB \ Bty, 'RelTol', 1e-8);
        end

        function choleskyFactorIsUpperTriangular(tc)
            rng(8);
            B = randn(20, 3);
            BtB = B' * B;
            Bty = B' * randn(20, 1);
            info = qf_info(BtB, Bty);

            R = info.chol;
            tc.verifyEqual(R' * R, BtB, 'RelTol', 1e-8);
            % upper triangular: lower part is zero
            tc.verifyEqual(tril(R, -1), zeros(3), 'AbsTol', 1e-10);
        end

        function nonPositiveDefiniteFails(tc)
            BtB = [1 2; 2 1];   % indefinite -> chol errors
            Bty = [1; 1];
            info = qf_info(BtB, Bty);
            tc.verifyFalse(info.fullrank);
            tc.verifyTrue(isnan(info.qf));
        end

        function rankDeficientGuardFails(tc)
            % Diagonal Cholesky ratio > 1e3 triggers the fullrank=false guard.
            BtB = diag([1, 1e8]);
            Bty = [1; 1];
            info = qf_info(BtB, Bty);
            tc.verifyFalse(info.fullrank);
            tc.verifyTrue(isnan(info.qf));
        end

        function getInvCholInvertsFactor(tc)
            rng(9);
            B = randn(15, 3);
            BtB = B' * B;
            Bty = B' * randn(15, 1);
            info = qf_info(BtB, Bty);
            info = info.get_inv_chol();
            tc.verifyEqual(info.inv_chol, info.chol \ eye(info.dim), 'RelTol', 1e-8);
            % chol * inv_chol == I
            tc.verifyEqual(info.chol * info.inv_chol, eye(3), 'AbsTol', 1e-8);
        end
    end
end
