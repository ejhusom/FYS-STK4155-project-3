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
    """Wrapper class for Scikit-Learn's boosting methods."""

    def __init__(self, 
            X_train, X_test, y_train, y_test,
            method='adaboost',
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

        if method == 'adaboost':
            self.adabooster()
        elif method == 'gradientboost':
            self.gradientbooster()
        elif method == 'xgboost':
            self.xgbooster()
        else:
            print('Provide boost method.')
            sys.exit(1)

        print('--------------------------------------------'
        print(f'Boost method: {self.method}')


    def boost_analysis():
        np.random.seed(2020)


        boost = Boost(method='xgboost')
        boost.clf.fit(X_train, y_train)
        accuracy = cross_validate(boost.clf, X_test, y_test, cv=10)['test_score']
        print(accuracy)

        y_pred = boost.clf.predict(X_test)

        plot_confusion_matrix(y_test, y_pred)


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
    data = ActivityData(dirname='data/activity/', subjects=[1,2,3])
    X, y = data.get_matrices()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test = scale_data(X_train, X_test, scaler='standard')
    analysis = AnalyzeBoost(X_train, X_test, y_train, y_test)
