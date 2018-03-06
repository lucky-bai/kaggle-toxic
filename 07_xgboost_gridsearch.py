
# coding: utf-8

# In[1]:


import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
import sklearn.svm
import xgboost


# In[ ]:


data = pd.read_csv('train_features.csv')
data_test = pd.read_csv('test_features.csv')


# In[3]:


SEED = 63467
data_train, data_valid = sklearn.model_selection.train_test_split(data, random_state = SEED, test_size = 0.1)


# Columns are either target or features

# In[15]:


target_cols = data_train.columns[1:7]
feature_cols = data_train.columns[7:]


# Let's try some classifiers

# In[18]:



# In[ ]:

target_name = 'toxic'
Y = data_train[target_name]
X = data_train[feature_cols]


# Grid search for XGBoost
for max_depth, n_estimators in [(9, 750), (10, 850), (11, 950), (12, 1000), (15, 1300)]:
    model = xgboost.XGBClassifier(
        n_jobs = 32,
        seed = SEED,
        max_depth = max_depth,
        n_estimators = n_estimators
    )
    model.fit(X, Y)
    predictions = model.predict_proba(data_valid[feature_cols])[:,1]
    actual = data_valid[target_name]
    cv_score = sklearn.metrics.roc_auc_score(actual, predictions)
    print(max_depth, n_estimators, cv_score)


# Run on test set and do submission
