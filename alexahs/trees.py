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

def test_trees(bias_var=False, random_forest=False):
    # dataset = load_boston()
    # X = dataset.data
    # y = dataset.target


    X, y = skl.datasets.make_regression(n_samples = 1000, n_features=4, n_informative=2)

    # print(X)
    # print(y)

    X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X, y,
            test_size = 0.2)


    scaler = skl.preprocessing.StandardScaler()

    scaler.fit_transform(X_train)
    scaler.transform(X_test)


    if bias_var:

        maxdepth = 10
        n_boostraps = 100

        error = np.zeros(maxdepth)
        bias = np.zeros(maxdepth)
        variance = np.zeros(maxdepth)
        polydegree = np.zeros(maxdepth)

        simpletree = DecisionTreeRegressor(max_depth=3)
        simpletree.fit(X_train, y_train)
        simpleprediction = simpletree.predict(X_test)
        for degree in range(1,maxdepth):
            model = DecisionTreeRegressor(max_depth=degree)
            y_pred = np.empty((y_test.shape[0], n_boostraps))
            for i in range(n_boostraps):
                x_, y_ = skl.utils.resample(X_train, y_train)
                model.fit(x_, y_)
                y_pred[:, i] = model.predict(X_test)#.ravel()

            polydegree[degree] = degree
            error[degree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
            bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
            variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )
            print('Polynomial degree:', degree)
            print('Error:', error[degree])
            print('Bias^2:', bias[degree])
            print('Var:', variance[degree])
            print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

        mse_simpletree = np.mean( np.mean((y_test - simpleprediction)**2))
        plt.xlim(1,maxdepth)
        plt.plot(polydegree, error, label='MSE simple tree')
        plt.plot(polydegree, mse_simpletree, label='MSE for Bootstrap')
        plt.plot(polydegree, bias, label='bias')
        plt.plot(polydegree, variance, label='Variance')
        plt.legend()
        plt.show()


    if random_forest:

        model = DecisionTreeRegressor(max_depth = 5)
        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        err_train = skl.metrics.mean_squared_error(pred_train, y_train)
        err_test = skl.metrics.mean_squared_error(pred_test, y_test)

        print('median target value',np.median(y_test))
        print('===one tree===')
        print('train:', err_train)
        print('test:', err_test)


        rfc = skl.ensemble.RandomForestRegressor(n_estimators = 100,
                                                 criterion='mse',
                                                 max_depth=5,
                                                 max_features='sqrt',
                                                 bootstrap=True)

        rfc.fit(X_train, y_train)

        pred_train = rfc.predict(X_train)
        pred_test = rfc.predict(X_test)

        err_train = skl.metrics.mean_squared_error(pred_train, y_train)
        err_test = skl.metrics.mean_squared_error(pred_test, y_test)

        print('===forest===')
        print('train:', err_train)
        print('test:', err_test)




def parameter_tuning_trees(model, X, y, param_name, param_range, y_label='Accuracy score'):
    train_scores, test_scores = skl.model_selection.validation_curve(
                                                estimator = model, X=X, y=y,
                                                param_name=param_name,
                                                param_range=param_range,
                                                cv=5)


    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_scores_mean, label='Training Score')
    plt.plot(param_range, test_scores_mean, label='Cross-validation Score')
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2)
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2)
    plt.legend(loc='best')
    plt.xlabel(param_name)
    plt.ylabel(y_label)
    plt.show()



def main():

    X, y = skl.datasets.load_wine(return_X_y = True)



    X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X, y, test_size = 0.2)

    scaler = skl.preprocessing.MinMaxScaler(feature_range=(-1, 1))

    scaler.fit_transform(X_train)
    scaler.transform(X_test)


    rfc = skl.ensemble.RandomForestClassifier(n_estimators = 200,
                                             criterion='gini',
                                             max_depth=3,
                                             max_features='sqrt',
                                             bootstrap=True)
    #

    parameter_tuning_trees(rfc, X_train, y_train, param_name='n_estimators', param_range=range(1, 1000, 200))





if __name__ == "__main__":
    main()
