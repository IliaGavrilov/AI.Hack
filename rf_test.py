import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


train_10 = pd.read_csv('train_data_mega_set_10_base.csv')

y_train_10 = train_10['proba']
X_train_10 = train_10.drop('proba',axis=1)


X_train, X_test, y_train, y_test = train_test_split(X_train_10, y_train_10.values, test_size=0.33, random_state=42)

rfc = RandomForestClassifier(n_jobs = -1,n_estimators = 20)
rfc.fit(X_train,y_train)
predict_proba = rfc.predict_proba(X_test)[:, 1]


from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif


X_train, X_test, y_train, y_test = train_test_split(X_train_10_best_cols_keys, y_train_10.values, test_size=0.33, random_state=42)

# rfc = RandomForestClassifier(n_jobs = -1,n_estimators = 20)

print(cross_val_score(rfc, X_train_10_best_cols_keys, y_train_10, scoring='roc_auc_score').mean())
