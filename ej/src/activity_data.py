#!/usr/bin/env python3
# ============================================================================
# File:     Activity data
# Author:   Erik Johannes Husom
# Created:  2019-11-19
# ----------------------------------------------------------------------------
# Description:
#
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



    def load_data(self):

        # Array for holding the data for all chosen subjects
        self.data = np.empty((0,6))
        
        for s in self.subjects:
            raw_data = np.loadtxt(open(os.path.join(dirname, str(s) + '.csv'),
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

    
    def remove_time_dimension(self, window=52):

        n_samples = len(self.data[s][:,0]) // window
        print(n_samples)
        subject_data = np.zeros((n_samples, len(self.data[s][0]) + 3))
        for n in range(n_samples):
            subject_data[n,:-3] = np.mean(self.data[s][n*window:(n+1)*window],
                    axis=0)
            subject_data[n,-3:] = np.std(self.data[s][n*window:(n+1)*window,1:4], axis=0)

        self.smp_data[s] = subject_data

        print(self.smp_data)
    

    def add_features(self, window=100):
        """Add features to model.
        
        Added features:
        - Mean of x, y, z acceleration (3)
        - Std of x, y, z acceleration (3)
        - Minmax of x, y, z acceleration (3)
        - Velocity in x, y, z direction (3)
        - Magnitude of acceleration?


        """

        n_new_features = np.shape(self.data)[1] + 12
        self.new_data = np.empty((np.shape(self.data)[0], n_new_features))



        

if __name__ == '__main__':
    dirname = 'data/activity/'
    data = ActivityData(dirname, subjects=[1,2])
    # data.explore_data()
    # data.simplify_features()
