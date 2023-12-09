from typing import Callable
from unittest import TestCase, TestLoader, TestSuite

import numpy as np
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility

from homeworks.lasso.ISTA import (
    convergence_criterion,
    loss,
    step,
)


class TestLassoISTA(TestCase):
    @visibility("visible")
    @partial_credit(.25)
    def test_loss_no_reg(self, set_score: Callable[[int], None]):
        # Generate data
        X = np.array([[0.2, 0.3], [0.4, 0.3], [0.1, -0.5], [-0.4, 0.5]])
        y = np.array([0.1, 0.2, -0.1, 0.2])
        weight = np.array([0.5, 1.5])  # L2 norm is sqrt(2.5)
        bias = 0.1
        _lambda = 0

        expected = 1.0575

        actual = loss(X, y, weight, bias, _lambda)
        np.testing.assert_almost_equal(actual, expected, decimal=4)
        set_score(0.25)

    @visibility("visible")
    @partial_credit(0.25)
    def test_loss_reg(self, set_score: Callable[[int], None]):
        # Generate data
        X = np.array([[0.2, 0.3], [0.4, 0.3], [0.1, -0.5], [-0.4, 0.5]])
        y = np.array([0.1, 0.2, -0.1, 0.2])
        weight = np.array([-0.5, 1.5])  # L1 norm is 2
        bias = 0.1
        _lambda = 1

        expected = 3.2275

        actual = loss(X, y, weight, bias, _lambda)
        np.testing.assert_almost_equal(actual, expected, decimal=4)
        set_score(0.25)

    @visibility("visible")
    @partial_credit(0.25)
    def test_convergence_criterion_pass(self, set_score: Callable[[int], None]):
        # Generate data
        _delta = 0.05
        weight_old = np.array([0.2, 0.3, 0.4, 0.5])
        weight = np.array([0.21, 0.325, 0.38, 0.46])
        bias_old = 0.4
        bias = 0.42

        expected = True  # Reg takes lambda * L2-norm^2

        actual = convergence_criterion(
            np.copy(weight), np.copy(weight_old), bias, bias_old, convergence_delta=_delta
        )
        assert actual == expected
        set_score(0.25)

    @visibility("visible")
    @partial_credit(0.25)
    def test_convergence_criterion_fail(self, set_score: Callable[[int], None]):
        _delta = 0.035
        weight_old = np.array([0.2, 0.3, 0.4, 0.5])
        weight = np.array([0.21, 0.325, 0.38, 0.455])
        bias_old = 0.4
        bias = 0.42

        expected = False  # Reg takes lambda * L2-norm^2

        actual = convergence_criterion(
            np.copy(weight), np.copy(weight_old), bias, bias_old, convergence_delta=_delta
        )
        assert actual == expected
        set_score(0.25)

    @visibility("visible")
    @partial_credit(0.25)
    def test_step(self, set_score: Callable[[int], None]):
        # Generate data
        X = np.array([[0.2, 0.3], [0.4, 0.3], [0.1, -0.5], [-0.4, 0.5]])
        y = np.array([0.1, 0.2, -0.1, 0.2])
        weight = np.array([0.5, 1.5])  # L2 norm is sqrt(2.5)
        bias = 0.2
        _lambda = 0.1
        eta = 0.01

        expected_weight = [0.4954, 1.4807]
        expected_bias = 0.171

        actual_weight, actual_bias = step(X, y, np.copy(weight), bias, _lambda, eta)
        np.testing.assert_almost_equal(actual_bias, expected_bias, decimal=4)
        np.testing.assert_array_almost_equal(actual_weight, expected_weight, decimal=4)
        set_score(0.25)

    @visibility("visible")
    @partial_credit(0.25)
    def test_step_zeros(self, set_score: Callable[[int], None]):
        # Generate data
        X = np.array([[0.2, 0.3], [0.4, 0.3], [0.1, -0.5], [-0.4, 0.5]])
        y = np.array([0.1, 0.2, -0.1, 0.2])
        weight = np.array([0.5, 1.5])  # L2 norm is sqrt(2.5)
        bias = 0.1
        _lambda = 0.5
        eta = 0.5

        expected_weight = [0.0, 0.195]
        expected_bias = -0.95

        actual_weight, actual_bias = step(X, y, np.copy(weight), bias, _lambda, eta)
        np.testing.assert_almost_equal(actual_bias, expected_bias, decimal=4)
        np.testing.assert_array_almost_equal(actual_weight, expected_weight, decimal=4)
        set_score(0.25)


# Create a Suite for this problem
suite_lasso_ista = TestLoader().loadTestsFromTestCase(
    TestLassoISTA
)

LassoISTATestSuite = TestSuite([suite_lasso_ista])
