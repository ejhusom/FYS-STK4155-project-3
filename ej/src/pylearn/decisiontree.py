#!/usr/bin/env python3
# ============================================================================
# File:     decisiontree.py
# Created:  2019-11-14
# ----------------------------------------------------------------------------
# Description:
#
# ============================================================================
import numpy as np


class Node():

    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None


class DecisionTreeRegression():
    """Decision tree for regression problems.

    Parameters
    ----------

    Attributes
    ----------

    """


    def __init__(self, max_depth, labels):
        
        self.max_depth = max_depth
        self.labels = labels




    def fit(X, y):
        pass



    def predict():
        pass
