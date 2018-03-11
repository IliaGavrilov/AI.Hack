import math
import operator

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

test_10 = pd.read_csv('test_data_mega_set_10_base_cat.csv', index_col='year_month', header=1).iloc[1:, :]
# test_12 = pd.read_csv('test_data_mega_set_12_base_cat.csv', index_col='year_month', header=1).iloc[1:, :]


def get_X_y(train_10):
    train_cols = train_10.columns.tolist()
    churn_col = train_cols[len(train_cols) - 2]

    y_train_10 = train_10[churn_col]
    X_train_10 = train_10.drop(churn_col, axis=1)
    print(y_train_10.value_counts())

    return (X_train_10, y_train_10)

X_test_10, y_test_10 = get_X_y(test_10)

X_test_10_norm = (X_test_10 - X_test_10.mean()) / (X_test_10.max() - X_test_10.min())

from scipy.spatial.distance import pdist
dist = pdist(X_test_10_norm, 'euclidean')

import matplotlib.pyplot as plt
plt.hist(dist, 500, color='green', alpha= 0.5)

