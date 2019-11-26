#!/usr/bin/env python3
# ============================================================================
# File:     boosting.py
# Author:   Erik Johannes Husom
# Created:  2019-11-20
# ----------------------------------------------------------------------------
# Description:
# Performing several boosting methods:
# - AdaBoost
# - Gradient boost
# - XGBoost
#
# Dataset:
# Labels are codified by numbers
# 1: Working at Computer
# 2: Standing Up, Walking and Going up\down stairs
# 3: Standing
# 4: Walking
# 5: Going Up\Down Stairs
# 6: Walking and Talking with Someone
# 7: Talking while Standing
#
# Simplified targets:
# 1: Working at computer 
# 2: Standing (combined class 3 and 7)
# 3: Walking (combined class 4 and 6)
# 4: Going up/down stairs (class 5)
# Class 2 is removed
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.stats import randint as sp_randint
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
        train_test_split, 
        cross_validate,
        GridSearchCV,
        RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import scikitplot as skplt
import xgboost as xgb
import sys
import os
import time

from ActivityData import ActivityData


def scale_data(train_data, test_data, scaler='standard'):
    """Scale train and test data.

    Parameters
    ----------
    train_data : array
        Train data to be scaled. Used as scale reference for test data.
    test_data : array
        Test data too be scaled, with train scaling as reference.
    scaler : str, default='standard'
        Options: 'standard, 'minmax'.
        Specifies whether to use sklearn's StandardScaler or MinMaxScaler.


    Returns
    -------
    train_data : array
        Scaled train data.
    test_data : array
        Scaled test data.

    """

    if scaler == 'standard':
        sc = StandardScaler()
    elif scaler == 'minmax':
        sc = MinMaxScaler()
    else:
        print('Scaler must be "standard" or "minmax"!')
        return None

    train_data = sc.fit_transform(train_data)
    test_data = sc.transform(test_data)

    return train_data, test_data


def plot_confusion_matrix(y_test, y_pred, analysis_id=None):
    """Plotting confusion matrix of a classification model.
    
    Parameters
    ----------
    y_test : array
        Target values from the test set.
    y_pred : array
        Predicted targets of the test set.
    analysis_id : str, default=None
        ID to be used when saving the plot. If None, a timestamp is used.
        
    """

    ax = skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True, cmap='Blues',
            title=' ')

    # Fixing cropped top and bottom
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    # Save figure with unique filename
    if analysis_id == None:
        analysis_id = time.strftime('%Y%m%d-%H%M')
    plt.savefig(analysis_id + '-confusionmatrix.pdf')

    plt.show()


def report(results, n_top=3):
    """Utility function for printing results from grid search.
    
    Taken from Scikit-Learn's documentation:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search
    """

    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


class AnalyzeBoost():
    """Analyzing the performance of three different boosting methods:
    - AdaBoost,
    - Gradient boost,
    - XGBoost.

    Parameters
    ----------
    X_train : array
      Features of the training set.
    X_test : array
      Features of the test set.
    y_train : array
      Targets of the training set.
    y_test : array
      Targets of the test set.
    method : str
        Boosting method to analyze.
    seed : float
        Random seed.
    n_estimators : int
    learning_rate : float
    max_depth : int
    verbose : boolean
        If True, printouts from the process are provided.

    Attributes
    ----------
    attribute : float
       Description.


    """

    def __init__(self, 
            X_train, X_test, y_train, y_test,
            method='xgboost',
            seed=0,
            n_estimators=100,
            learning_rate=0.5,
            max_depth=3,
            verbose=True):


        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.method = method
        self.seed = seed
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.verbose = verbose

        self.time_id = time.strftime('%Y%m%d-%H%M%S')

        if self.verbose:
            print('-----------------------')
            print(f'Time: {self.time_id}')
            print(f'Number of training samples: {np.shape(self.X_train)[0]}')
            print(f'Number of test samples: {np.shape(self.X_test)[0]}')
            print(f'Method: {method}')


        if self.method == 'adaboost':
            self.base_estimator = DecisionTreeClassifier()
            self.clf = AdaBoostClassifier(base_estimator=self.base_estimator)
            self.max_depth_str = 'base_estimator__max_depth'
        elif self.method == 'gradientboost':
            self.clf = GradientBoostingClassifier()
            self.max_depth_str = 'max_depth'
        elif self.method == 'xgboost':
            self.clf = xgb.XGBClassifier()
            self.max_depth_str = 'max_depth'
        else:
            print('Provide boost method.')
            sys.exit(1)



    def fit(self):
        parameters = {
                'n_estimators': self.n_estimators, 
                'learning_rate': self.learning_rate,
                self.max_depth_str: self.max_depth}

        self.clf.set_params(**parameters)
        self.clf.fit(self.X_train, self.y_train)

        # Save model
        pickle.dump(self.clf, open(
            self.time_id + '-' + self.method + '-fit.pkl', 'wb'))
            


    def predict(self):

        self.y_pred = self.clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_pred, self.y_test)
        print(f'Test accuracy score: {np.around(accuracy, decimals=3)}')

        plot_confusion_matrix(self.y_test, self.y_pred,
                analysis_id=self.time_id)


    def gridsearch(self, parameters=None, cv=2, load_search=None):
        """Performing a grid search for optimal parameters.

        Parameters
        ----------
        parameters : dict
            Dictionary with the parameters to be tested in the grid search.
        cv : int
            Number of folds in the cross-validation.
        load_search : pickle dump
            The search model from a potential previous grid search, to avoid
            doing a new grid search. If None, a new grid search is performed.


        """
        
        if load_search is None:
            if parameters is None:
                parameters =  [
                        {'learning_rate': [1, 0.5],
                         'n_estimators': [100, 150, 200],
                         self.max_depth_str: [5, 7]}
                ]

            self.search = GridSearchCV(self.clf, param_grid=parameters, cv=cv,
                    n_jobs=3)
            self.search.fit(X_train, y_train)

            # Save model
            pickle.dump(self.search, open(
                self.time_id + '-' + self.method + '-search.pkl', 'wb'))

        else:
            self.search = pickle.load(open(load_search, 'rb'))

        # Save results from grid search and print to terminal
        cv_results = pd.DataFrame(self.search.cv_results_) 
        cv_results.to_csv(f'{self.time_id}-gridsearch.csv')
        report(self.search.cv_results_)

        # Overwriting parameters to the best parameters found by search
        self.learning_rate = self.search.best_params_['learning_rate']
        self.n_estimators = self.search.best_params_['n_estimators']
        self.max_depth = self.search.best_params_[self.max_depth_str]




if __name__ == '__main__':
    """Analyzing boosting methods when used for human activity recognition
    (HAR).

    The program analyzes to different cases:

    - Case 1: All 15 subjects are loaded in to a dataset, and then a random
      split is performed to get training and test data.
    - Case 2: 12 subjects are used as training data, and the remaining 3
      subjects are used for the test set.

    """


    np.random.seed(2020)

    try:
        case = sys.argv[1]
    except:
        print('Give case number (1 or 2) as command line argument.')
        sys.exit(1)

    if case == '1':
        """Training/test split is done on all subjects."""

        data_file = 'activity_data_preprocessed_case1.npy'

        # Preprocess data if not already done
        if not os.path.exists(data_file):
            data = ActivityData(dir='data/activity/')
            data.output_to_npy(data_file)
        
        data = np.load(data_file)
        X = data[:,:-1]
        y = data[:,-1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    elif case == '2':
        """Training and test sets contain different subjects."""

        train_data_file = 'activity_data_preprocessed_case2_training.npy'
        test_data_file = 'activity_data_preprocessed_case2_test.npy'

        # Preprocess data if not already done
        if not os.path.exists(train_data_file):
            train_data = ActivityData(dir='data/activity/',
                    subjects=list(range(1,13)))
            train_data.output_to_npy(train_data_file)
            test_data = ActivityData(dir='data/activity/',
                    subjects=list(range(13,16)))
            test_data.output_to_npy(test_data_file)
        
        train_data = np.load(train_data_file)
        test_data = np.load(test_data_file)
        X_train = train_data[:,:-1]
        X_test = test_data[:,:-1]
        y_train = train_data[:,-1]
        y_test = test_data[:,-1]

    else:
        print('Choose case 1 or 2.')
        sys.exit(1)

    X_train, X_test = scale_data(X_train, X_test, scaler='standard')

    analysis = AnalyzeBoost(X_train, X_test, y_train, y_test,
            method='xgboost',
            n_estimators=150,
            learning_rate=0.5,
            max_depth=5)

    # analysis.gridsearch()
    analysis.fit()
    analysis.predict()

    # analysis = AnalyzeBoost(X_train, X_test, y_train, y_test, method='xgboost')
    # analysis.fit()
    # analysis = AnalyzeBoost(X_train, X_test, y_train, y_test, method='gradientboost')
    # analysis = AnalyzeBoost(X_train, X_test, y_train, y_test, method='adaboost')
