import pandas as pd
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

train_10 = pd.read_csv('train_data_mega_set_10_base_cat.csv',index_col = 'year_month',header=1).iloc[1:,:]
train_cols = train_10.columns.tolist()
churn_col = train_cols[len(train_cols)-2]

y_train_10 = train_10[churn_col]
X_train_10 = train_10.drop(churn_col,axis=1)

model = ExtraTreesClassifier()
model.fit(X_train_10, y_train_10)
# display the relative importance of each attribute


feature_importances = model.feature_importances_

len(train_10)