import numpy as np
import os, glob

class ActivityData:

    def __init__(self, dir, subjects = list(range(1,16))):
        self.freq = 52
        self.dir = dir
        self.subjects = subjects
        self.data_is_loaded = False

    def load_data(self):

        data_temp = {}

        i = 1
        for s in self.subjects:
            print('Loading subject %i ...' %s)
            raw_data = np.loadtxt(open(os.path.join(self.dir, str(s) + '.csv'),
                        'rb'), delimiter=',')[:,1:]

            data_temp[i] = raw_data
            i += 1


        self.data = data_temp[1]
        # for i in range(2, len(filenames)+1):
        if len(self.subjects) > 1:
            for i in range(2, len(self.subjects)+1):
                self.data = np.concatenate((self.data, data_temp[i]), axis=0)

        self.data_is_loaded = True
        print('Loading complete.')


    def split_classes(self):

        self.classes = {}

        for i in range(1, 8):
            self.classes[i] = self.data[np.where(self.data[:,-1] == np.float64(i))[0],:]




    def create_features(self, which_class):

        n_rows_raw = self.classes[which_class].shape[0]

        n_rows = 2*(n_rows_raw // self.freq) - 1
        step = int(self.freq / 2)

        acc_mean = np.zeros((n_rows-1, 3))
        acc_std = np.zeros((n_rows-1, 3))
        minmax = np.zeros((n_rows-1, 3))
        mean_vel = np.zeros((n_rows-1, 3))
        magnitude = np.zeros((n_rows-1, 1))
        frequency = np.zeros((n_rows-1, 3))


        for i in range(n_rows - 1):
            start = i*step
            stop = start + 2*step

            interval_acc = self.classes[which_class][start:stop,:-1]

            frequency[i,:] = np.linalg.norm(np.fft.rfft(interval_acc, axis=0), axis=0, ord=2)
            acc_mean[i,:] = np.mean(interval_acc, axis=0)
            acc_std[i,:] = np.std(interval_acc, axis=0)
            magnitude[i] = np.linalg.norm(acc_mean[i, :], ord=2)
            minmax[i,:] = np.max((interval_acc), axis=0) - np.min((interval_acc), axis=0)


        dt = 1
        for i in range(1, n_rows - 1):
            mean_vel[i,:] = mean_vel[i-1,:] + acc_mean[i,:]*dt


        targets = np.ones(n_rows - 1)*which_class


        features = np.concatenate((acc_mean,
                            acc_std,
                            minmax,
                            mean_vel,
                            magnitude,
                            frequency), axis=1)

        return features, targets


    def get_feature_matrix(self):
        if self.data_is_loaded == False:
            self.load_data()
        self.split_classes()


        X, y = self.create_features(1)

        for i in range(2, 8):
            X_temp, y_temp = self.create_features(i)
            X = np.concatenate((X, X_temp), axis=0)
            y = np.concatenate((y, y_temp), axis=0)


        return X, y.astype(int)

    def output_to_csv(self, filename='activity_data_preprocessed.csv'):
        X, y = self.get_feature_matrix()
        df = pd.DataFrame(np.concatenate((X, y), axis=1))
        df.to_csv(filename)
        print('Saved data to %s' %filename)

    def output_to_npy(self, filename='activity_data_preprocessed.npy'):
        X, y = self.get_feature_matrix()
        y = y.reshape((y.shape[0], 1))
        np.save(filename, np.concatenate((X, y), axis=1))
        print('Saved data to %s' %filename)
