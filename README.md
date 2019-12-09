# FYS-STK4155 - Project 3: Classifying human activity using trees and ensembles

This project uses decision trees and ensemble methods to classify human
activity based on accelerometer data. For details of the methods and results of
the project, read the report found in this repository. For details about the
source code, read below.

---

![](har-mario-figure.png)

## Source code

The source code of this project consists of the following files:

- `ActivityData.py`: Preprocessing of the raw dataset.
- `Boosting.py`: Machine learning using boosting algorithms.
- `Trees.py`: Machine learning using a single decision tree, bagging and random
  forest.


### Running boosting analysis

The usage of the script `Boosting.py` is done via command line arguments, and
requires som additional information. The script is executed by running 

```sh
python Boosting.py [case] [method] [n_estimators] [learning_rate] [max_depth]
```

The options are as follows:

- `[case]`:
    - `1`: All subjects mixed in training and test set(setting 1).
    - `2`: Separated subjects in test set (setting 2).
    - `3`: Plot feature importance. This requires that both case 1 and 2 has been
      analyzed by all methods, since that analysis includes evaluation of
      feature importance.
- `[method]`:
    - `adaboost`: AdaBoost.
    - `gradientboost`: Gradient boosting.
    - `xgboost`: XGBoost.
- `[n_estimators]` (optional): Number of estimators when performing boosting. Usually
  from 50 to 250.
- `[learning_rate]` (optional): Learning rate. Usually from 1 down to 0.001.
- `[max_depth]` (optional): Max tree depth of the base estimator. Usually from 3 to 15.

If `[n_estimators]`, `[learning_rate]` and `[max_depth]` is not given, the
script will run a grid search to tune these parameters. The range of the grid
search is hard-coded in the script. This is because running the grid search requires
extensive computational resources, and we wanted to restrict the potential run
time. To change this, go into the function `gridsearch()` and change the
default values of `parameters`.

### Running trees analysis

By running `python Trees.py`, the analysis of decision tree, bagging and random
forest will be run. The default behaviour is to run the analysis for setting 1
(all subjects mixed in training and test set). To run setting 2 (separate
subjects in test set), simply comment out the lines marked with `SETTING 2` in
the last section of the file.


