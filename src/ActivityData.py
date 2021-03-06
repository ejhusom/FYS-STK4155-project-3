import numpy as np
import os, glob
import pandas as pd
import sklearn as skl


class ActivityData:
    """
    Class for creating descriptive features from accelerometer readings in the
    "Activity Recognition from Single Chest-Mounted Accelerometer" data set from the UCI website.

    Parameters
    ----------
    dir : string
        Folder directory containing the data files.
    subjects : array
        List of data from subjects to load and preprocess.
    simplify : boolean
        Whether to join similar activities and simplify the target features
        (reduces no. of classes from 7 to 4)

    Methods
    -------
    load_data(self)
        Loads subject data and stores it in self.data (dict).
    split_classes(self)
        Separates and stores observations of target features in self.classes (dict)
    create_features(self, which_class)
        Creates descriptive features of one target class from windows
        of 52 accelerometer readings:
            -Mean acceleration (x, y, z)
            -Standard deviation of acceleration (x, y, z)
            -MinMax value of acceleration(x, y, z)
            -Mean velocity(x, y, z)
            -Magnitude of acceleration
            -2-norm of the frequency of acceleration(x, y, z)
    get_feature_matrix(self)
        Returns a single feature matrix and target vector of all subjects.
    output_to_csv(self, filename)
        Saves the feature matrix and target vector to a single .csv file.
    output_to_npy(self, filename)
        Saves the feature matrix and target vector to a single .npy file.

    """
    def __init__(self, dir, subjects=list(range(1, 16)), simplify=False):
        self.freq = 52
        self.dir = dir
        self.subjects = subjects
        self.data_is_loaded = False
        self.n_targets = 7
        self.simplify = simplify

    def load_data(self):

        #Temporary storage of the data
        data_temp = {}

        #Load data files
        i = 1
        for s in self.subjects:
            print("Loading subject %i ..." % s)
            raw_data = np.loadtxt(
                open(os.path.join(self.dir, str(s) + ".csv"), "rb"), delimiter=","
            )[:, 1:]

            # Simplify target features
            if self.simplify:
                self.n_targets = 4


                raw_data = np.delete(
                    raw_data, np.where(raw_data[:, -1] == 2)[0], axis=0
                )

                raw_data[np.where(raw_data[:, -1] == 3)[0], -1] = 2
                raw_data[np.where(raw_data[:, -1] == 7)[0], -1] = 2
                raw_data[np.where(raw_data[:, -1] == 4)[0], -1] = 3
                raw_data[np.where(raw_data[:, -1] == 6)[0], -1] = 3
                raw_data[np.where(raw_data[:, -1] == 5)[0], -1] = 4
            #end if

            data_temp[i] = raw_data
            i += 1

        #end for

        #Collect data from all subjects
        self.data = data_temp[1]
        if len(self.subjects) > 1:
            for i in range(2, len(self.subjects) + 1):
                self.data = np.concatenate((self.data, data_temp[i]), axis=0)

        self.data_is_loaded = True
        print("Loading complete.")

    def split_classes(self):

        self.classes = {}

        #Split observations of target features
        for i in range(1, self.n_targets + 1):
            self.classes[i] = self.data[
                np.where(self.data[:, -1] == np.float64(i))[0], :
            ]

    def create_features(self, which_class):


        n_rows_raw = self.classes[which_class].shape[0]
        n_rows = 2 * (n_rows_raw // self.freq) - 1
        step = int(self.freq / 2)

        #Initialize descriptive feature matrices
        acc_mean = np.zeros((n_rows - 1, 3))
        acc_std = np.zeros((n_rows - 1, 3))
        minmax = np.zeros((n_rows - 1, 3))
        mean_vel = np.zeros((n_rows - 1, 3))
        magnitude = np.zeros((n_rows - 1, 1))
        frequency = np.zeros((n_rows - 1, 3))

        #Loop through all observations and calculate features within window
        for i in range(n_rows - 1):
            start = i * step
            stop = start + 2 * step

            interval_acc = self.classes[which_class][start:stop, :-1]

            frequency[i, :] = np.linalg.norm(
                np.fft.rfft(interval_acc, axis=0), axis=0, ord=2
            )
            acc_mean[i, :] = np.mean(interval_acc, axis=0)
            acc_std[i, :] = np.std(interval_acc, axis=0)
            magnitude[i] = np.linalg.norm(acc_mean[i, :], ord=2)
            minmax[i, :] = np.max((interval_acc), axis=0) - np.min(
                (interval_acc), axis=0
            )

        #Calculate mean velocity
        dt = 1
        for i in range(1, n_rows - 1):
            mean_vel[i, :] = mean_vel[i - 1, :] + acc_mean[i, :] * dt

        #create feature matrix and target vector
        targets = np.ones((n_rows - 1, 1)) * which_class
        features = np.concatenate(
            (acc_mean, acc_std, minmax, mean_vel, magnitude, frequency), axis=1
        )

        return features, targets

    def get_feature_matrix(self):

        if self.data_is_loaded == False:
            self.load_data()
        self.split_classes()

        #Get feature matrix
        X, y = self.create_features(1)
        for i in range(2, self.n_targets + 1):
            X_temp, y_temp = self.create_features(i)
            X = np.concatenate((X, X_temp), axis=0)
            y = np.concatenate((y, y_temp), axis=0)

        temp_matrix = np.concatenate((X, y), axis=1)

        #Shuffle observations
        np.random.shuffle(temp_matrix)

        return temp_matrix[:, :-1], temp_matrix[:, -1].astype(int)

    def output_to_csv(self, filename="activity_data_preprocessed.csv"):
        X, y = self.get_feature_matrix()
        df = pd.DataFrame(np.concatenate((X, y), axis=1))
        df.to_csv(filename)
        print("Saved data to %s" % filename)

    def output_to_npy(self, filename="activity_data_preprocessed.npy"):
        X, y = self.get_feature_matrix()
        y = y.reshape((y.shape[0], 1))
        np.save(filename, np.concatenate((X, y), axis=1))
        print("Saved data to %s" % filename)
