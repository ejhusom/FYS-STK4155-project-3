import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from ActivityData import *

"""
1: Working at Computer
2: Standing Up, Walking and Going up\down stairs
3: Standing
4: Walking
5: Going Up\Down Stairs
6: Walking and Talking with Someone
7: Talking while Standing
"""




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



def main():
    dir = "data/activity/"
    # filenames = os.listdir(dir)
    # filenames.remove('README')
    # filename = filenames[0]


    # data = np.loadtxt(dir + filename, delimiter=',')[:, 1:]

    data = ActivityData(dir)
    X, y = data.get_feature_matrix()

    print(X.shape)
    print(y.shape)


    # dat = ActivityData(data)
    # act.split_and_join()
    # act.get_window_indices()
    # print(aData.joined_data)
    # act.create_features()

    # n_rows, rows_per_activity, rows_downsamp =  get_downsample_size(data)

    # X, y = load_activity_data(dir, which_person=0)
    # test_voting()



    # plot_activity(dir, 0)


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
