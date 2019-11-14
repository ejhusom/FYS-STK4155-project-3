#!/usr/bin/env python3
# ============================================================================
# File:     test_DecisionTreeRegressor.py
# Author:   Erik Johannes Husom
# Created:  2019-11-14
# ----------------------------------------------------------------------------
# Description:
#
# ============================================================================


def generate_data():
    """Generate random data set to be used for regression testing."""

    X = np.linspace(-3, 3, n).reshape(-1, 1)
    y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)

    return X, y

