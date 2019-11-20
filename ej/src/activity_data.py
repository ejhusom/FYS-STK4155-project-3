#!/usr/bin/env python3
# ============================================================================
# File:     Activity data
# Author:   Erik Johannes Husom
# Created:  2019-11-19
# ----------------------------------------------------------------------------
# Description:
#
# --- Labels are codified by numbers
#     --- 1: Working at Computer
#     --- 2: Standing Up, Walking and Going up\down stairs
#     --- 3: Standing
#     --- 4: Walking
#     --- 5: Going Up\Down Stairs
#     --- 6: Walking and Talking with Someone
#     --- 7: Talking while Standing
# ============================================================================
import sys
import os
import matplotlib.pyplot as plt
import numpy as np


class ActivityData():

    def __init__(self, dirname, subjects=list(range(1,16))):

        self.dirname = dirname
        self.subjects = subjects
        self.n_subjects = len(subjects)
        
        self.load_data()
        self.add_features()



    def load_data(self):

        # Array for holding the data for all chosen subjects
        self.data = np.empty((0,6))
        
        for s in self.subjects:
            raw_data = np.loadtxt(open(os.path.join(self.dirname, str(s) + '.csv'),
                    'rb'), delimiter=',')
            n_samples = np.shape(raw_data)[0]
            n_features = np.shape(raw_data)[1]

            # New array that contains the data, with an added column that
            # contains the subject id
            subject_data = np.empty((n_samples, n_features + 1))
            subject_data[:,1:] = raw_data
            subject_data[:,0] = s

            # Remove rows with 0 as target
            zero_rows = np.where(subject_data[:,-1] == 0)
            subject_data = np.delete(subject_data, zero_rows, axis=0)

            print('------------------------------')
            print(f'Subject {s} loaded.')
            print(f'Number of samples: {n_samples}')

            # Find number of samples with certain activity
            for i in range(1,8):
                count = np.size(np.where(subject_data[:,-1] == i))
                print(f'Target {i}: {count} samples')

            # Adding the data to the main array
            self.data = np.concatenate((self.data, subject_data), axis=0)

            print(f'Number of zero rows removed: {np.size(zero_rows)}')

        print('------------------------------')
        print('Complete data set:')
        for i in range(1,8):
            count = np.size(np.where(self.data[:,-1] == i))
            print(f'Target {i}: {count} samples')

    


    def explore_data(self, subject=1, smp=False):

        plt.figure()

        if smp:
            data = self.smp_data[subject]
            t = data[:,0]
            x_std = data[:,5]
            y_std = data[:,6]
            z_std = data[:,7]

            plt.scatter(t, x_std, s=0.01)
            plt.scatter(t, y_std, s=0.01)
            plt.scatter(t, z_std, s=0.01)
        else:
            data = self.data[subject]

        t = data[:,0]
        x = data[:,1]
        y = data[:,2]
        z = data[:,3]
        activity = data[:,4]

        plt.scatter(t, x, label='x', s=0.01)
        plt.scatter(t, y, label='y', s=0.01)
        plt.scatter(t, z, label='z', s=0.01)
        # plt.scatter(t, activity, c=activity, s=1)

        plt.show()

    

    def add_features(self, window=52):
        """Add features to model.

        Old columns (6):
        0: ID
        1: sample number
        2: x acc
        3: y acc
        4: z acc
        5: activity

        Added columns (13):
        0: ID
        1: mean sample number
        2: mean x acc
        3: mean y acc
        4: mean z acc
        5: activity
        6: std x acc
        7: std y acc
        8: std z acc
        9: minmax x acc
        10: minmax y acc
        11: minmax z acc
        12: x vel
        13: y vel
        14: z vel
        15: acc magnitude
        """


        n_new_features = np.shape(self.data)[1] + 3
        n_new_samples = len(self.data[:,0]) // window
        self.new_data = np.empty((n_new_samples, n_new_features))

        for n in range(n_new_samples):
            self.new_data[n,:-3] = np.mean(self.data[n*window:(n+1)*window], axis=0)
            self.new_data[n,-3:] = np.std(self.data[n*window:(n+1)*window,2:5], axis=0)
            self.new_data[n,5] = np.asarray(self.new_data[n,5], dtype=int)



    def get_matrices(self):

        X_columns = [2,3,4,6,7,8]
        y_columns = [5]
        X = self.new_data[:, X_columns]
        y = np.asarray(np.ravel(self.new_data[:, y_columns]), dtype=int)

        return X, y

        

if __name__ == '__main__':
    dirname = 'data/activity/'
    data = ActivityData(dirname, subjects=[1])
    # data.explore_data()
    data.add_features()
