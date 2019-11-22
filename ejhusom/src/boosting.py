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
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
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
    """Plotting confusion matrix of a classification model."""

    ax = skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True, cmap='Blues',
            title=' ')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    # Save figure with unique filename
    if analysis_id == None:
        analysis_id = time.strftime('%Y%m%d-%H%M')
    plt.savefig(analysis_id + '-confusionmatrix.pdf')

    plt.show()

def report(results, n_top=3):
    """Utility function from Scikit-Learn's documentation.
    scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search
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

    def __init__(self, 
            X_train, X_test, y_train, y_test,
            method='xgboost',
            search_method=None,
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
        self.search_method = search_method
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

        self.analyze()



    def analyze(self):
        if self.method == 'adaboost':
            self.adabooster()
        elif self.method == 'gradientboost':
            self.gradientbooster()
        elif self.method == 'xgboost':
            self.xgbooster()
        else:
            print('Provide boost method.')
            sys.exit(1)

        if self.search_method == 'grid':
            self.gridsearch()
        elif self.search_method == 'random':
            self.randomsearch()
        else:
            parameters = {
                    'n_estimators': self.n_estimators, 
                    'learning_rate': self.learning_rate,
                    'max_depth': self.max_depth}

            self.clf.set_params(**parameters)

            self.clf.fit(self.X_train, self.y_train)
            accuracy = cross_validate(self.clf, self.X_test, self.y_test, cv=10)['test_score']
            print(f'Accuracy: {np.around(accuracy, decimals=3)}')

            self.y_pred = self.clf.predict(self.X_test)

            plot_confusion_matrix(self.y_test, self.y_pred,
                    analysis_id=self.time_id)


    def gridsearch(self, parameters=None, cv=5):
        
        if parameters is None:
            parameters =  [
                    {'learning_rate': [1, 0.5, 0.1, 0.05, 0.01],
                     'n_estimators': [50, 100, 150, 200]}
            ]
            if self.method != 'adaboost':
                parameters['max_depth']: [3, 5, 7, 9]

        self.search = GridSearchCV(self.clf, param_grid=parameters, cv=cv,
                n_jobs=3)
        self.search.fit(X_train, y_train)

        # Save results from grid search and print to terminal
        cv_results = pd.DataFrame(self.search.cv_results_) 
        cv_results.to_csv(f'{self.time_id}-gridsearch.csv')
        report(self.search.cv_results_)

        best_learning_rate = self.search.best_params_['learning_rate']
        best_n_estimators = self.search.best_params_['n_estimators']
        if

    
    def randomsearch(self, parameters=None, n_iter=10, cv=10):
        
        if parameters is None:
            parameters =  [
                    {'learning_rate': [1, 0.5, 0.1],#, 0.05, 0.01],
                     'n_estimators': sp_randint(10,100)}#, 100, 150]}
            ]
            if self.method != 'adaboost':
                parameters['max_depth']: [3, 5, 7, 9]

        self.search = RandomizedSearchCV(self.clf,
                param_distributions=parameters, n_iter=n_iter, cv=cv)
        self.search.fit(X_train, y_train)
        report(self.search.cv_results_)



    def adabooster(self):

        self.clf = AdaBoostClassifier()


    def gradientbooster(self):

        self.clf = GradientBoostingClassifier()


    def xgbooster(self):
        
        self.clf = xgb.XGBClassifier()


if __name__ == '__main__':
    np.random.seed(2020)

    if not os.path.exists('activity_data_preprocessed.npy'):
        data = ActivityData(dir='data/activity/', n_files=1)
        data.output_to_npy()

    data = np.load('activity_data_preprocessed.npy')
    X = data[:,:-1]
    y = data[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test = scale_data(X_train, X_test, scaler='standard')

    analysis = AnalyzeBoost(X_train, X_test, y_train, y_test,
            method='adaboost',
            search_method='grid')

    # analysis = AnalyzeBoost(X_train, X_test, y_train, y_test, method='xgboost')
    # analysis = AnalyzeBoost(X_train, X_test, y_train, y_test, method='gradientboost')
    # analysis = AnalyzeBoost(X_train, X_test, y_train, y_test, method='adaboost')
