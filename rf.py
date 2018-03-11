import math
import operator

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier

from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def get_base():
    train_10 = pd.read_csv('train_data_mega_set_10_base.csv', index_col='year_month', header=1).iloc[1:, :]
    train_12 = pd.read_csv('train_data_mega_set_12_base.csv', index_col='year_month', header=1).iloc[1:, :]
    test_10 = pd.read_csv('test_data_mega_set_10_base.csv', index_col='year_month', header=1).iloc[1:, :]
    test_12 = pd.read_csv('test_data_mega_set_12_base.csv', index_col='year_month', header=1).iloc[1:, :]
    return (train_10, train_12, test_10, test_12)


def get_base_cat():
    train_10 = pd.read_csv('train_data_mega_set_10_base_cat.csv', index_col='year_month', header=1).iloc[1:, :]
    train_12 = pd.read_csv('train_data_mega_set_12_base_cat.csv', index_col='year_month', header=1).iloc[1:, :]
    test_10 = pd.read_csv('test_data_mega_set_10_base_cat.csv', index_col='year_month', header=1).iloc[1:, :]
    test_12 = pd.read_csv('test_data_mega_set_12_base_cat.csv', index_col='year_month', header=1).iloc[1:, :]
    return (train_10, train_12, test_10, test_12)



def base_rf(train_10, train_12, test_10, test_12, submition_name, rf_10=False, rf_12=False):
    train_cols = train_10.columns.tolist()
    churn_col = train_cols[len(train_cols) - 2]

    test_10 = test_10[test_10[churn_col] == 1]

    X_train_10, y_train_10 = get_X_y(train_10)
    X_train_10_balanced, y_train_10_balanced = sm.fit_sample(X_train_10, y_train_10)

    X_train_12, y_train_12 = get_X_y(train_12)
    X_train_12_balanced, y_train_12_balanced = sm.fit_sample(X_train_12, y_train_12)

    X_test_10, y_test_10 = get_X_y(test_10)
    X_test_12, y_test_12 = get_X_y(test_12)

    rf_10 = rf_10 if rf_10 != False else RandomForestClassifier(n_estimators=50, n_jobs=-1)
    rf_12 = rf_12 if rf_12 != False else RandomForestClassifier(n_estimators=50, n_jobs=-1)

    rf_10.fit(X_train_10_balanced, y_train_10_balanced)
    rf_12.fit(X_train_12_balanced, y_train_12_balanced)

    score(rf_10,X_train_10, y_train_10)
    score(rf_12, X_train_12, y_train_12)

    predict_proba_10 = rf_10.predict_proba(X_test_10)[:, 1]
    predict_proba_12 = rf_12.predict_proba(X_test_12)[:, 1]

    preds_10 = pd.Series(data=predict_proba_10, index=y_test_10.index)
    preds_12 = pd.Series(data=predict_proba_12, index=y_test_12.index)

    preds = preds_10.append(preds_12)

    preds.to_csv(submition_name + '.csv')

    return rf_10, rf_12, preds

sm = SMOTE()

def get_X_y(train_10):
    train_cols = train_10.columns.tolist()
    churn_col = train_cols[len(train_cols) - 2]

    y_train_10 = train_10[churn_col]
    X_train_10 = train_10.drop(churn_col, axis=1)
    print(y_train_10.value_counts())

    return (X_train_10, y_train_10)


def score(classifier, X, y):
    print(cross_val_score(classifier, X, y, scoring='roc_auc').mean())


def get_base(type='train', month='10'):
    train = pd.read_csv(type + '_data_mega_set_' + month + '_base.csv', index_col='year_month', header=1).iloc[1:, :]
    return train


def get_base_cat(type='train', month='10', best_cols_keys=False):
    train = pd.read_csv(type + '_data_mega_set_' + month + '_base_cat.csv', index_col='year_month', header=1).iloc[1:,:]
    if(best_cols_keys == False):
        X, y = get_X_y(train)
        model = ExtraTreesClassifier()
        model.fit(X, y)
        # display the relative importance of each attribute
        best_cols_num = int(math.ceil(math.log(len(X), 2)))
        columns_importance = dict(zip(X.columns.tolist(), list(model.feature_importances_)))
        best_cols = sorted(columns_importance.items(), key=operator.itemgetter(1))[-best_cols_num:]

        best_cols_keys = [t[0] for t in best_cols]
        train_cols = train.columns.tolist()[-2:]
        best_cols_keys = list(set(best_cols_keys).difference(set(train_cols))) + train_cols

    train = train[best_cols_keys]

    return train, best_cols_keys


# train_10,train_cols_keys_10 = get_base_cat('train', '10')
# train_12,train_cols_keys_12 = get_base_cat('train', '12')
# test_10,test_cols_keys_10 = get_base_cat('test', '10', train_cols_keys_10)
# test_12,test_cols_keys_10 = get_base_cat('test', '12', train_cols_keys_12)

train_10 = get_base('train', '10')
train_12 = get_base('train', '12')
test_10 = get_base('test', '10')
test_12 = get_base('test', '12')


rf_10 = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100,n_jobs=-1,max_depth=15,min_samples_leaf=3,min_samples_split=5), method='isotonic', cv=5)
rf_12 = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100,n_jobs=-1,max_depth=15,min_samples_leaf=3,min_samples_split=5), method='isotonic', cv=5)

rf_10, rf_12, rf_result = base_rf(train_10, train_12, test_10, test_12, 'submition_base_calibrated_balanced_tune_rf',rf_10,rf_12)



knn_10 = KNeighborsClassifier(RandomForestClassifier(n_estimators=200,n_jobs=-1,max_depth=30,min_samples_leaf=3,min_samples_split=5), method='isotonic', cv=5)
knn_12 = KNeighborsClassifier(RandomForestClassifier(n_estimators=200,n_jobs=-1,max_depth=30,min_samples_leaf=3,min_samples_split=5), method='isotonic', cv=5)

knn_10, knn_12, rf_result = base_rf(train_10, train_12, test_10, test_12, 'submition_base_calibrated_balanced_tune_knn', knn_10,knn_12)



xgd_10 = XGBClassifier()
xgd_12 = XGBClassifier()
#
xgd_10, xgd_12, xgd_result= base_rf(train_10, train_12, test_10, test_12, 'submition_base_calibrated_balanced_tune_xgd',xgd_10,xgd_12)

