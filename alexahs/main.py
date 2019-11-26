import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz #
from ActivityData import *
from functions import *

"""
1: Working at Computer
2: Standing Up, Walking and Going up\down stairs
3: Standing
4: Walking
5: Going Up\Down Stairs
6: Walking and Talking with Someone
7: Talking while Standing

1: Working at computer
2 = 3 + 7: Standing
3 = 4 + 6: Walking
5: going up/down stairs
"""





def main():
    # """
    #preprocess data
    dir = "data/activity/"

    # data = ActivityData(dir)
    # X, y = data.get_feature_matrix()



    data_train = ActivityData(dir, list(range(1, 13)))
    data_train.output_to_npy('data/activity_train.npy')
    data_test = ActivityData(dir, list(range(13, 16)))
    data_test.output_to_npy('data/activity_test.npy')
    # """


    data_train = np.load('data/activity_train.npy')
    data_test = np.load('data/activity_test.npy')


    X_train, y_train = data_train[:,:-1], data_train[:,-1]
    X_test, y_test = data_test[:,:-1], data_test[:,-1]


    # data = np.load('data/activity_data_full_preprocessed.npy')
    # X, y = data[:,:-1], data[:,-1]
    # X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X, y, test_size=0.2)

    scaler = skl.preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)




    #single tree
    # """
    param_name = 'max_depth'
    param_range = range(2, 20)

    analyze_simple_tree(X_train, X_test, y_train, y_test, param_name, param_range, x_label='Tree depth')
    # """
    #trees
    """
    parameters = [
        {'criterion': ['entropy'], 'n_estimators': [100, 150, 200], 'max_depth': [3, 5, 7, 9, 11]},
        # {'criterion': ['gini'], 'n_estimators': [10, 50, 100, 150], 'max_depth': [3, 5, 7]},
    ]

    analyze_trees(X_train, X_test, y_train, y_test, parameters, plotting=True, save_to_file=False, n_jobs=-1)
    """



    # validation_curve(model, X_train, y_train, 'n_estimators', param_range=range(1, 400, 50))


    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)







def test_voting():
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    log_clf = LogisticRegression(solver="liblinear", random_state=42)
    rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
    svm_clf = SVC(gamma="auto", random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='hard')

    voting_clf.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score

    print('===HARD VOTING===')
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

    log_clf = LogisticRegression(solver="liblinear", random_state=42)
    rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
    svm_clf = SVC(gamma="auto", probability=True, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='soft')
    voting_clf.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score

    print('===SOFT VOTING===')
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))



if __name__ == '__main__':
    main()
