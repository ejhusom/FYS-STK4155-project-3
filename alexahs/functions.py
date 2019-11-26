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
plt.style.use('ggplot')



def plot_activity(dir, which_person):
    filenames = os.listdir(dir)
    filenames.remove('README')
    filename = filenames[which_person]

    data = np.loadtxt(dir + filename, delimiter=',')[:, 1:]


    activities = ['working at computer',
    'Standing Up, Walking and Going up/down stairs',
    'standing', 'walking',
    'Going Up/Down Stairs',
    'Walking and Talking with Someone',
    'Talking while Standing']
    for i in range(7):


        inds = np.where(data[:,-1] == i+1)[0]

        x, y, z = data[inds, 0], data[inds, 1], data[inds, 2]

        n = range(len(x))

        plt.scatter(n, x, label='x', s=1)
        plt.scatter(n, y, label='y', s=1)
        plt.scatter(n, z, label='z', s=1)
        plt.legend()
        plt.title(activities[i])
        plt.show()

def log_results(results):
    pass



def validation_curve(model, X, y, param_name, param_range, x_label=None, y_label='Accuracy score'):
    train_scores, validation_scores = skl.model_selection.validation_curve(
                                                estimator=model, X=X, y=y,
                                                param_name=param_name,
                                                param_range=param_range,
                                                cv=5, n_jobs=-1, verbose=5)


    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(test_scores, axis=1)
    validation_scores_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_scores_mean, label='Training Score')
    plt.plot(param_range, test_scores_mean, label='Cross-validation Score')
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2)
    plt.fill_between(param_range, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.2)
    plt.legend(loc='best')
    if x_label == None:
        plt.xlabel(param_name)
    else:
        plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def grid_search(model, X, y, parameters, filename, save_to_file = False, n_jobs = -1):

    clf = skl.model_selection.GridSearchCV(estimator=model,
                                           param_grid=parameters,
                                           cv=5, verbose=6, n_jobs=n_jobs)

    clf.fit(X, y)

    best_params = clf.best_params_


    if save_to_file:
        results = clf.cv_results_
        df = pd.DataFrame(results)
        df.to_csv(filename + '_cv_results_best_index_' + str(clf.best_index_) + '.csv')

    print('Best model parameters: %s' %best_params)
    print('Validation score: %5.3f' %clf.best_score_)


    return best_params


def analyze_trees(X_train, X_test, y_train, y_test, parameters, max_features='sqrt', plotting = True, save_to_file = False, n_jobs = -1):
    """
    for bagging: set max_features=None
    """

    if max_features == None:
        filename = 'bagging'
    else:
        filename = 'random_forest'


    """
    model_cv = skl.ensemble.RandomForestClassifier(max_features=max_features, bootstrap=True)

    params = grid_search(model_cv, X_train, y_train, parameters, filename=filename, save_to_file=True, n_jobs = n_jobs)
    """

    params = {'criterion': 'entropy', 'n_estimators': 150, 'max_depth': 13}

    model_test = skl.ensemble.RandomForestClassifier(
                                                    n_estimators = params['n_estimators'],
                                                    criterion = params['criterion'],
                                                    max_depth = params['max_depth'],
                                                    max_features = max_features)

    model_test.fit(X_train, y_train)
    y_pred = model_test.predict(X_test)

    print('Test score: %5.3f' %skl.metrics.accuracy_score(y_pred, y_test))

    if plotting:
        ax = skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True, cmap='Blues',
                title=' ')
        bottom, top = ax.get_ylim()

        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.show()


def analyze_simple_tree(X_train, X_test, y_train, y_test, param_name, param_range, plotting=True, x_label=None, y_label='Accuracy score'):



    model_cv = skl.tree.DecisionTreeClassifier()

    train_scores, validation_scores = skl.model_selection.validation_curve(
                                                estimator=model_cv, X=X_train, y=y_train,
                                                param_name=param_name,
                                                param_range=param_range,
                                                cv=5, n_jobs=-1, verbose=5)
    #



    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)


    best_score_ind = np.argmax(validation_scores_mean)
    params = [i for i in param_range]
    max_depth_opt = params[best_score_ind]

    model_test = skl.tree.DecisionTreeClassifier(max_depth=max_depth_opt)
    model_test.fit(X_train, y_train)
    y_pred = model_test.predict(X_test)




    print('===Scores===')
    print('Best depth: %i' %max_depth_opt)
    print('Validation score: %5.3f' %validation_scores_mean[best_score_ind])
    print('Test score: %5.3f' %skl.metrics.accuracy_score(y_pred, y_test))


    if plotting:
        #train vs validation scores
        plt.plot(param_range, train_scores_mean, label='Training Score')
        plt.plot(param_range, validation_scores_mean, label='Validation Score')
        plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.3)
        plt.fill_between(param_range, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.3)
        plt.legend(loc='best')
        if x_label == None:
            plt.xlabel(param_name)
        else:
            plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()


        #confusion matrix
        ax = skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True, cmap='Blues',
                title=' ')
        bottom, top = ax.get_ylim()

        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.show()
