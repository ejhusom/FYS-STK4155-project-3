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
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import scikitplot as skplt
import xgboost as xgb
import sys
import time

from activity_data import *


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
    plt.savefig('confusionmatrix-' + analysis_id + '.pdf')

    plt.show()



class AnalyzeBoost():

    def __init__(self, 
            scale=True, 
            method='xgboost',
            seed=0,
            n_estimators=100,
            learning_rate=0.5,
            max_depth=3,
            verbose=True):


        self.data = ActivityData(dirname='data/activity/', subjects=[1,2,3])
        self.X, self.y = self.data.get_matrices()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

        if scale:
            self.X_train, self.X_test = scale_data(self.X_train, self.X_test, scaler='standard')

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
            print(f'Scaled: {scale}')
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


        self.clf.fit(self.X_train, self.y_train)
        accuracy = cross_validate(self.clf, self.X_test, self.y_test, cv=10)['test_score']
        print(f'Accuracy: {accuracy}')

        self.y_pred = self.clf.predict(self.X_test)

        plot_confusion_matrix(self.y_test, self.y_pred,
                analysis_id=self.time_id)


    def adabooster(self, base_estimator=DecisionTreeClassifier()):

        self.clf = AdaBoostClassifier(
                base_estimator=base_estimator,
                n_estimators=self.n_estimators, 
                algorithm='SAMME.R', 
                learning_rate=self.learning_rate,
                random_state=self.seed)
        
    
    def gradientbooster(self):

        self.clf = GradientBoostingClassifier(
                max_depth=self.max_depth, 
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate)



    def xgbooster(self):
        
        self.clf = xgb.XGBClassifier()

if __name__ == '__main__':
    np.random.seed(2020)

    # data = ActivityData(dirname='data/activity/', subjects=[1,2,3])
    # X, y = data.get_matrices()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # X_train, X_test = scale_data(X_train, X_test, scaler='standard')


    analysis = AnalyzeBoost(method='xgboost')
    analysis = AnalyzeBoost(method='gradientboost')
    analysis = AnalyzeBoost(method='adaboost')
