import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
import sklearn as skl
import sklearn.ensemble
import sklearn.datasets
import sklearn.model_selection
import sklearn.tree
import sklearn.metrics
import scikitplot as skplt
import time
from ActivityData import *

plt.style.use("ggplot")


class Trees:
    """
    Class for analyzing hyper parameters of decision trees, random forests and
    bagging using Scikit Learn.

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
    max_features : string
        Number of total predictive features to use when growing trees.
        For a bagging classifier, use None.
    n_jobs : int
        Number of processes to run in parallel.


    Methods
    -------
    log_results(self, results, filename)
        Saves results to a text file.
    analyze_tree(self, parameters, plotting, save_results)
        Uses cross-validation to determine best set of hyper parameters for
        a model, tests the best model on the test data, saves results to file
        and plots the confusion matrix.
    analyze_simple_tree(self, param_name, param_range, plotting, save_results)
        Does a 1D parameter search for a single decision tree and plots the
        validation curve of the results and tests the best model on the test data.
    """


    def __init__(self, X_train, X_test, y_train, y_test, max_features = 'sqrt', n_jobs=-1):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.base_model = skl.ensemble.RandomForestClassifier(n_estimators=50, max_depth=3, bootstrap=True, n_jobs=n_jobs, max_features=max_features)

    def log_results(self, results, filename):
        filename += "_results.txt"
        outfile = open(filename, "w")

        #Write results to file
        for key in results:
            print(str(key) + ": " + str(results[key]))
            outfile.write(str(key) + ": " + str(results[key]) + "\n")

        outfile.close()
        print("Results logged to %s" % filename)

    def analyze_tree(self, parameters, plotting=True, save_results=True):

        #Perform grid search
        search = skl.model_selection.GridSearchCV(
            estimator=self.base_model,
            param_grid=parameters,
            return_train_score=True,
            cv=5,
            verbose=5,
            n_jobs=self.n_jobs,
        )

        search.fit(self.X_train, self.y_train)

        #Grid search scores
        train_score = search.cv_results_["mean_train_score"][search.best_index_]
        val_score = search.best_score_
        params_opt = search.best_params_

        #Test model on test data
        model_test = skl.ensemble.RandomForestClassifier(
            n_estimators=params_opt["n_estimators"],
            criterion=params_opt["criterion"],
            max_depth=params_opt["max_depth"],
            max_features=self.max_features,
        )
        model_test.fit(self.X_train, self.y_train)
        y_pred = model_test.predict(self.X_test)


        #Print some stats
        parameter_space = []
        for key in parameters:
            parameter_space.append(str(key))

        results = {}
        results.update({"search values": parameter_space})
        results.update(params_opt)
        results.update(
            {
                "Train score": train_score,
                "Validation score": val_score,
                "Test score": skl.metrics.accuracy_score(y_pred, self.y_test),
            }
        )

        print(results)

        id = time.strftime("%Y%m%d-%H%M%S")

        if self.max_features == None:
            filename = "results/" + id + "_bagging"
        else:
            filename = "results/" + id + "_randomforest"

        #Save results to file and plot
        if save_results:
            self.log_results(results, filename)

        if plotting:
            ax = skplt.metrics.plot_confusion_matrix(
                self.y_test, y_pred, normalize=True, cmap="Blues", title=" "
            )
            bottom, top = ax.get_ylim()

            ax.set_ylim(bottom + 0.5, top - 0.5)
            if save_results:
                plt.savefig(filename + "_confusion_matrix.pdf")
            plt.show()

    def simple_tree(self, param_name='max_depth', param_range=range(1, 20), plotting=True, save_results=True):

        model_cv = skl.tree.DecisionTreeClassifier()

        #Perform search
        train_scores, validation_scores = skl.model_selection.validation_curve(
            estimator=model_cv,
            X=self.X_train,
            y=self.y_train,
            param_name=param_name,
            param_range=param_range,
            cv=5,
            n_jobs=self.n_jobs,
            verbose=5,
        )

        #Calculate mean score values
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        validation_scores_mean = np.mean(validation_scores, axis=1)
        validation_scores_std = np.std(validation_scores, axis=1)

        best_score_ind = np.argmax(validation_scores_mean)
        params = list(param_range)
        max_depth_opt = params[best_score_ind]

        #Test model with optimal parameters on test data
        model_test = skl.tree.DecisionTreeClassifier(max_depth=max_depth_opt)
        model_test.fit(self.X_train, self.y_train)
        y_pred = model_test.predict(self.X_test)


        results = {
            "Best depth": max_depth_opt,
            "Train score": train_scores_mean[best_score_ind],
            "Validation score": validation_scores_mean[best_score_ind],
            "Test score": skl.metrics.accuracy_score(y_pred, self.y_test),
        }

        id = time.strftime("%Y%m%d-%H%M%S")
        filename = "results/" + id + "_simple_tree"
        if save_results:
            log_results(results, filename)

        if plotting:
            # train vs validation scores
            plt.plot(param_range, train_scores_mean, label="Training Score")
            plt.plot(param_range, validation_scores_mean, label="Validation Score")
            plt.fill_between(
                param_range,
                train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std,
                alpha=0.3,
            )
            plt.fill_between(
                param_range,
                validation_scores_mean - validation_scores_std,
                validation_scores_mean + validation_scores_std,
                alpha=0.3,
            )
            plt.legend(loc="best")
            plt.xlabel(param_name)
            plt.ylabel('Accuracy score')
            if save_results:
                plt.savefig(filename + "_tree_depth_opt.pdf")
            plt.show()

            # confusion matrix
            ax = skplt.metrics.plot_confusion_matrix(
                self.y_test, y_pred, normalize=True, cmap="Blues", title=" "
            )
            bottom, top = ax.get_ylim()

            ax.set_ylim(bottom + 0.5, top - 0.5)
            plt.show()

if __name__ == '__main__':
    dir_data = "data/activity/"

    """MIXED SUBJECTS(SETTING 1)"""
    data = ActivityData(dir_data)
    X, y = data.get_feature_matrix()
    X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X, y, test_size=0.2)
    """END SETTING 1"""

    """DIFFERENT SUBJECTS FOR TRAINING AND TEST(SETTING 2)"""
    # data_train = ActivityData(dir_data, list(range(1, 13)), simplify=False)
    # data_test = ActivityData(dir_data, list(range(13, 16)), simplify=False)
    # X_train, y_train = data_train[:, :-1], data_train[:, -1]
    # X_test, y_test = data_test[:, :-1], data_test[:, -1]
    """END SETTING 2"""


    scaler = skl.preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    parameters = [
        {'criterion': ['entropy', 'gini'],
         'n_estimators': [50, 100, 150, 200, 250, 300],
          'max_depth': [3, 5, 7, 9, 11, 13, 15]},
    ]


    random_forest = Trees(X_train, X_test, y_train, y_test, max_features='sqrt')
    random_forest.analyze_tree(parameters)
    bagging = Trees(X_train, X_test, y_train, y_test, max_features=None)
    bagging.analyze_tree(parameters)
    decision_tree = Trees(X_train, X_test, y_train, y_test)
    decision_tree.analyze_simple_tree(param_name='max_depth')
