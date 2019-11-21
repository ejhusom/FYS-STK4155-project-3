import numpy as np
import os, glob

class ActivityData:

    def __init__(self, dir, remove_files = None):
        self.freq = 52
        self.dir = dir
        self.remove_files = remove_files

    def load_data(self):
        dir_list = os.listdir(self.dir)
        filenames = []
        for file in dir_list:
            if file.endswith('.csv'):
                filenames.append(file)

        data_temp = {}

        i = 1
        for filename in filenames:
            print('Loading %s ...' %filename)
            raw_data = np.loadtxt(self.dir + filename, delimiter=',')[:, 1:]
            data_temp[i] = raw_data
            i += 1


        self.data = data_temp[1]
        for i in range(2, len(filenames)+1):
            self.data = np.concatenate((self.data, data_temp[i]), axis=0)

        print('Loading complete.')


    def split_classes(self):

        self.classes = {}

        for i in range(1, 8):
            self.classes[i] = self.data[np.where(self.data[:,-1] == np.float64(i))[0],:]



    def create_features(self, which_class=5):

        n_rows_raw = self.classes[which_class].shape[0]

        n_rows = 2*(n_rows_raw // self.freq) - 1
        step = int(self.freq / 2)

        acc_mean = np.zeros((n_rows-1, 3))
        acc_std = np.zeros((n_rows-1, 3))
        minmax = np.zeros((n_rows-1, 3))
        mean_vel = np.zeros((n_rows-1, 3))
        magnitude = np.zeros(n_rows-1)


        for i in range(n_rows - 1):
            start = i*step
            stop = start + 2*step
            interval_acc = self.classes[which_class][start:stop,:-1]

            acc_mean[i,:] = np.mean(interval_acc, axis=0)
            acc_std[i,:] = np.std(interval_acc, axis=0)
            magnitude[i] = np.linalg.norm(acc_mean[i, :], ord=2)

            minmax[i,:] = np.max((interval_acc), axis=0) - np.min((interval_acc), axis=0)


        dt = 1
        for i in range(1, n_rows - 1):
            mean_vel[i,:] = mean_vel[i-1,:] + acc_mean[i,:]*dt


        target = np.ones((n_rows - 1, 1))*which_class

        magnitude = magnitude.reshape(n_rows - 1, 1)

        feature_matrix = np.concatenate((acc_mean,
                            acc_std,
                            minmax,
                            mean_vel,
                            magnitude,
                            target), axis=1)


        features = feature_matrix[:,:-1]
        targets = feature_matrix[:,-1]

        return features, targets



    def get_feature_matrix(self):
        self.load_data()
        self.split_classes()


        X, y = self.create_features(1)

        for i in range(2, 8):
            X_temp, y_temp = self.create_features(i)
            X = np.concatenate((X, X_temp), axis=0)
            y = np.concatenate((y, y_temp), axis=0)


        return X, y.astype(int)
