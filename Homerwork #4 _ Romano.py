
# coding: utf-8

# # Homework #4 

# In[1]:


# Description of the project for me to refer to.
# Use a model that considers a family’s observable attributes to classify them and predict their level of need.


# # Part 1 | Import Packages & the Data Set

# In[2]:


# First I need to import various packages to get set up for the project
import datetime
import gc
import numpy as np
import os
import operator
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import describe
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb
import xgboost as xgb


# In[3]:


# import the data set | 
train = pd.read_csv("/Users/hunterromano/Desktop/train.csv")

# look at data to make sure it was imported and correct
train.head(5)


# # Part 2 | Clean & Explore the Data

# In[4]:


# I want a full list of the attributes in the set
# I can understand what data points I have and how to move forward
train.dtypes


# In[5]:


# I want a better understanding of the values 
train.describe()


# In[6]:


# More views
train.head().transpose()


# In[7]:


#shape of data
print('training_data:',train.shape)


# # Take care of missing values

# In[8]:


train.isnull().values.any()


# In[9]:


train.info()


# In[10]:


train.isnull().values.sum(axis=0)


# In[11]:


train_describe = train.describe()
train_describe


# In[12]:


train.isnull().values.sum(axis=0)


# # Look into the target variables

# In[13]:


plt.figure(figsize=(15, 8))
plt.hist(train.Target.values, bins=4)
plt.title('Histogram - Target Counts', fontsize=35)
plt.xlabel('Count', fontsize=28)
plt.ylabel('Target', fontsize=28)
plt.show()


# In[14]:


plt.title("Distribution of the Target", fontsize=30)
sns.distplot(train['Target'].dropna(),color='blue', kde = True, bins=100)
plt.show()


# In[15]:


sns.set_style("whitegrid")
ax = sns.violinplot(x=train.Target.values)
plt.show()


# In[16]:


plt.title("Distribution of log(target)")
sns.distplot(np.log1p(train['Target']).dropna(),color='blue', kde=True, bins=100)
plt.show()


# In[17]:


sns.set_style("whitegrid")
ax = sns.violinplot(x=np.log(1+train.Target.values))
plt.show()


# In[18]:


np.unique(train.Target.values)


# In[19]:


columns_functional = train.columns[1:-1]


# In[20]:


columns_functional


# In[21]:


y = train['Target'].values-1


# In[22]:


# I want to bring in my test data to clean it at the same time.
test = pd.read_csv("/Users/hunterromano/Desktop/test.csv")


# In[23]:


test.head()


# In[24]:


train_test_df = pd.concat([train[columns_functional], test[columns_functional]], axis=0)
# Identify the columns that contain categorical data
categorical_columns = [f_ for f_ in train_test_df.columns if train_test_df[f_].dtype == 'object']


# In[26]:


# Take a look at them
categorical_columns


# In[28]:


# Now I need to label my different columns
for col in categorical_columns:
    le = LabelEncoder()
    print(col)
    le.fit(train_test_df[col].astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    del le


# In[29]:


def dprint(*args, **kwargs):
    print("[{}] ".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) +         " ".join(map(str,args)), **kwargs)

id = 'Id'

target = 'Target'

df_all = pd.concat([train, test], axis=0)
cols = [f_ for f_ in df_all.columns if df_all[f_].dtype == 'object' and f_ != id]
print(cols)

for c in tqdm(cols):
    le = preprocessing.LabelEncoder()
    le.fit(df_all[c].astype(str))
    train_df[c] = le.transform(train[c].astype(str))
    test_df[c] = le.transform(test[c].astype(str))

    del le
gc.collect()

def extract_features(df):
    df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']
    df['rent_to_rooms'] = df['v2a1']/df['rooms']
    df['tamhog_to_rooms'] = df['tamhog']/df['rooms']

extract_features(train)
extract_features(test)


# In[30]:


# Find correlated varaibles with target
labels = []
values = []
for col in train.columns:
    if col not in ["Id", "Target"]:
        labels.append(col)
        values.append(np.corrcoef(train[col].values, train["Target"].values)[0,1])
corr_df = pd.DataFrame({'columns_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
 
corr_df = corr_df[(corr_df['corr_values']>0.20) | (corr_df['corr_values']<-0.20)]
ind = np.arange(corr_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(10,9))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.columns_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient", fontsize=27)
ax.set_title("Correlation coefficient each variables", fontsize=32)
plt.show()


# In[31]:


#Matrix of the variables with the highest correlation
temp = train[corr_df.columns_labels.tolist()]
correlation_map = temp.corr(method='pearson')
f, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(correlation_map, vmax=1., square=True, cmap=plt.cm.BrBG)
plt.title("Variables with Highest Correlation", fontsize=25)
plt.show()


# In[32]:


# Try forest hope data is good
train_undersampled=train.drop(train.query('Target == 4').sample(frac=.75).index)
train_undersampled


# In[33]:


X=train_undersampled.drop(['Id', 'idhogar', 'Target', 'edjefe', 'edjefa'], axis=1)
y=train_undersampled['Target']


# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[35]:


X_train.shape


# In[36]:


y_train.shape


# In[37]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[38]:


clf = RandomForestClassifier()
params={'n_estimators': list(range(40,61, 1))}
gs = GridSearchCV(clf, params, cv=5)


# In[39]:


gs.fit(X_train, y_train)


# In[41]:


# It looks like my data isn't clean enough to run a random forest unfortunately
# I tried cleaning it in various other ways but ended with the same result
#  I figured it would be better to try another model instead of just copying a perfect cleaning kernel that I didn't understand.
# So I did some research and found lightgbm, it didn't work for a while.
# I figured out that I had to install some new packages within the terminal to be able to run it.
# It took way longer than anticipated, but after several brew commands and pip installs my terminal is up-to-date


# In[40]:


# Now time to run my Predictive Model

# The following code begins the lightgbm
cnt = 0
p_buf = []
n_splits = 20
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=None)
err_buf = []   

cols_to_drop = [
    id, 
    target,
]
X = train.drop(cols_to_drop, axis=1, errors='ignore')
feature_names = list(X.columns)
X = X.fillna(0)
X = X.values
y = train[target].values

classes = np.unique(y)
dprint('Number of classes: {}'.format(len(classes)))
c2i = {}
i2c = {}
for i, c in enumerate(classes):
    c2i[c] = i
    i2c[i] = c

y_le = np.array([c2i[c] for c in y])

X_test = test.drop(cols_to_drop, axis=1, errors='ignore')
X_test = X_test.fillna(0)
X_test = X_test.values
id_test = test[id].values

dprint(X.shape, y.shape)
dprint(X_test.shape)

n_features = X.shape[1]

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'max_depth': -1,
    'num_leaves': 14,
    'learning_rate': 0.1,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': 8,
    'colsample_bytree': 0.89,
    'min_child_samples': 90,
    'subsample': 0.96,
    'lambda_l2': 1.0,
    'min_gain_to_split': 0,
    'num_class': len(np.unique(y)),
}


# In[42]:


#generate sampling with imblearn
sampler = RandomUnderSampler(random_state=314)
X, y = sampler.fit_sample(X, y)
y_le = np.array([c2i[c] for c in y])

for train_index, valid_index in kf.split(X, y):
    print('Fold {}/{}*{}'.format(cnt + 1, n_splits, n_repeats))
    params = lgb_params.copy() 

    lgb_train = lgb.Dataset(
        X[train_index], 
        y_le[train_index], 
        feature_name=feature_names,
        )
    lgb_train.raw_data = None

    lgb_valid = lgb.Dataset(
        X[valid_index], 
        y_le[valid_index],
        feature_name=feature_names,
        )
    lgb_valid.raw_data = None

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=99999,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=400, 
        verbose_eval=100, 
    )

    if cnt == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        for i in range(10):
            if i < len(tuples):
                print(i, tuples[i])
            else:
                break

        del importance, model_fnames, tuples

    p = model.predict(X[valid_index], num_iteration=model.best_iteration)

    err = f1_score(y_le[valid_index], np.argmax(p, axis=1), average='macro')

    dprint('{} F1: {}'.format(cnt + 1, err))

    p = model.predict(X_test, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p, dtype=np.float16)
    else:
        p_buf += np.array(p, dtype=np.float16)
    err_buf.append(err)

    cnt += 1

    del model, lgb_train, lgb_valid, p
    gc.collect


# In[43]:


err_mean = np.mean(err_buf)
err_std = np.std(err_buf)
print('F1 = {:.6f} +/- {:.6f}'.format(err_mean, err_std))
preds = p_buf/cnt


# In[55]:


print(preds)
preds = np.argmax(preds, axis = 1) +1
preds


# In[56]:


sample_submission = pd.read_csv("/Users/hunterromano/Desktop/sample_submission.csv")
sample_submission.head()


# In[57]:


sample_submission['Target'] = preds
sample_submission.to_csv('submission_{:.6f}.csv'.format(err_mean), index=False)
sample_submission.head()


# In[58]:


np.mean(preds)


# In[ ]:


# Ready to submit but it appears I cannot create a late submission

