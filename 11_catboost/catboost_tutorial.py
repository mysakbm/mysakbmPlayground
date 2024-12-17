import os
import numpy as np
import pandas as pd
import xgboost as xgb
import random
import catboost as cb
import category_encoders as ce

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as auc
from catboost import CatBoost,CatBoostClassifier, Pool

# In[] ---- Enviroment Settings ------------
pd.options.display.max_columns = 30

# In[] ---- Cat in dat dataset -------------------------------------------------
data = pd.read_csv("./data/cat-in-the-dat/train.csv")
train, test = train_test_split(data, test_size = 0.2, random_state = 1985)

target_train = train['target']
train.drop(['target', 'id'], axis=1, inplace=True)

target_test = test['target']
test.drop(['target', 'id'], axis=1, inplace=True)

# In[]  Catboost Model
categorical_features_indices = list(range(len(list(train))))[3:]
train_pool = Pool(train, target_train, cat_features=categorical_features_indices)
test_pool = Pool(test, cat_features = categorical_features_indices)


# In[] Pool: Pouziti funkce pool, ktera plni podobnou funkci jako dmatrix pro XGB.
model = CatBoostClassifier(iterations=20,
                           depth=2,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=True)

# train the model
model.fit(train_pool,
          eval_set = (test, target_test))

# make the prediction using the resulting model
preds_class = model.predict(test_pool)
preds_proba = model.predict_proba(test_pool)
print("class = ", preds_class)
print("proba = ", preds_proba)

auc(target_test, preds_class)


# In[] Bez pouziti funkce Pool to funguje taky:
model = cb.CatBoostClassifier(iterations = 20)

model.fit(train, target_train,
          eval_set = (test, target_test),
          cat_features = categorical_features_indices,
          use_best_model = True,
          verbose = True
          )

preds_class = model.predict(test)
preds_proba = model.predict_proba(test)
print("class = ", preds_class)
print("proba = ", preds_proba)

auc(target_test, preds_class)



# In[] --- Transformace promennych nejprv a az pak catboost

CBE_encoder = ce.CatBoostEncoder(verbose=1, cols = list(train.columns)[3:])
train_cbe = CBE_encoder.fit_transform(train, target_train)
test_cbe = CBE_encoder.transform(test)

model = cb.CatBoostClassifier(iterations = 20)

model.fit(train_cbe, target_train,
          eval_set = (test_cbe, target_test),
          # cat_features = [0, 1, 2],
          use_best_model = True,
          verbose = True
          )

preds_class = model.predict(test_cbe)
preds_proba = model.predict_proba(test_cbe)
print("class = ", preds_class)
print("proba = ", preds_proba)
auc(target_test, preds_class)


# In[] .....CLASS CatBoost - zjevne nejaky wrapper.
model = CatBoost()

# train the model
model.fit(train_pool,
          eval_set = (test, target_test))

# make the prediction using the resulting model
preds_class = model.predict(test_pool)
preds_proba = model.predict_proba(test_pool)
print("class = ", preds_class)
print("proba = ", preds_proba)

auc(target_test, preds_class)
