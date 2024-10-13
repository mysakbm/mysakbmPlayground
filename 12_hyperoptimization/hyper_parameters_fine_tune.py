# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import joblib
import random
import datetime

import matplotlib.pyplot as plt
import scikitplot as skplt

from tqdm import tqdm
from pathlib import Path

from hyperopt import hp, fmin, tpe, Trials
import optuna as optuna

import xgboost as xgb
import catboost as cb

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# In[] Aux Functions
def plot_lift_curve(y_val, y_pred, step=0.01):
    # Define an auxiliar dataframe to plot the curve
    aux_lift = pd.DataFrame()
    # Create a real and predicted column for our new DataFrame and assign values
    aux_lift['real'] = y_val
    aux_lift['predicted'] = y_pred
    # Order the values for the predicted probability column:
    aux_lift.sort_values('predicted', ascending=False, inplace=True)

    # Create the values that will go into the X axis of our plot
    x_val = np.arange(step, 1 + step, step)
    # Calculate the ratio of ones in our data
    ratio_ones = aux_lift['real'].sum() / len(aux_lift)
    # Create an empty vector with the values that will go on the Y axis our our plot
    y_v = []

    # Calculate for each x value its correspondent y value
    for x in x_val:
        num_data = int(
            np.ceil(x * len(aux_lift)))  # The ceil function returns the closest integer bigger than our number
        data_here = aux_lift.iloc[:num_data, :]  # ie. np.ceil(1.4) = 2
        ratio_ones_here = data_here['real'].sum() / len(data_here)
        y_v.append(ratio_ones_here / ratio_ones)

    return pd.DataFrame({"bin": x_val, "lift": y_v})

def _evaluate_model(target_test, y_test_pred):
    # Evaluate Model
    model_auc = auc(target_test, y_test_pred)

    fpr, tpr, thresholds = roc_curve(target_test, y_test_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    preds_class = np.zeros(len(y_test_pred))
    preds_class[y_test_pred > optimal_threshold] = 1

    tn, fp, fn, tp = confusion_matrix(target_test, preds_class).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # calcualte lift curve
    lift_df = plot_lift_curve(target_test, y_test_pred, step=0.1)

    return {"precision" : precision,
            "recall" : recall,
            "lift_df" : lift_df,
            "model_auc" : model_auc}

# In[]
samples = 20000
xgb_dummy = xgb.XGBClassifier(seed=47)
# generating the dataset of custom sample size
dummy = make_classification(n_samples=samples)

# splitting the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(dummy[0],
                                                    dummy[1],
                                                    test_size=0.2,
                                                    stratify=dummy[1],
                                                    random_state=42)

xgb_dummy.fit(X_train, y_train)

accuracy_score(y_test, xgb_dummy.predict(X_test))


# In[] XGBoost
model_name = "XGBoost"

default_params = {"booster" : "gbtree",
          "objective" : "binary:logistic",
          "eval_metric" : 'logloss',
          "learning_rate" : 0.2,
          "max_depth" : 6,
          "min_child_weight" : 2,
          "subsample" : 0.8,
          "colsample_bytree" : 0.8,
          "early_stopping_rounds" : 10,
          "random_state" : 1985}

model = xgb.XGBClassifier(n_estimators=600, **default_params)

model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          verbose=True)

y_test_pred = model.predict_proba(X_test, ntree_limit=model.best_iteration)
y_test_pred = y_test_pred[:,1]

eval_dict_xgb = _evaluate_model(y_test, y_test_pred)
print(eval_dict_xgb)

# In[] CatBoost

model_name = "CatBoost"

model = cb.CatBoostClassifier(loss_function='Logloss', verbose=True)

model.fit(X_train, y_train, eval_set = (X_test, y_test))

# make the prediction using the resulting model
y_test_pred = model.predict_proba(X_test)
y_test_pred = y_test_pred[:,1]

eval_dict_cb = _evaluate_model(y_test, y_test_pred)
print(eval_dict_cb)

# In[] HyperOpt
# https://medium.com/@rithpansanga/optimizing-xgboost-a-guide-to-hyperparameter-tuning-77b6e48e289d

model_name = "XGBoost"


# define models and parameters
def objective_function(params):
    print({k: round(v, 2) for k, v in params.items()})
    clf = xgb.XGBClassifier(n_estimators=200, objective='binary:logistic', **params)  # Assuming binary classification
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1985)
    score = -cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc").mean()  # Minimize negative score
    return score

# Search Space
max_depth_options = [3, 4, 5, 6, 7, 8]
learning_rate_options = [0.2, 0.1, 0.01, 0.001]
min_child_weight_options = [1, 2, 3, 5, 8, 10]

search_space = {
    'learning_rate': hp.choice('learning_rate', learning_rate_options),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.choice('min_child_weight', min_child_weight_options),
    'max_depth': hp.choice('max_depth', max_depth_options),
}

# Trials object to store results
trials = Trials()

# Run optimization
best = fmin(fn=objective_function, space=search_space, algo=tpe.suggest, trials=trials, max_evals=50)

best_params = best
best_params["max_depth"] = max_depth_options[best["max_depth"]]
best_params["learning_rate"] = learning_rate_options[best["learning_rate"]]
best_params["min_child_weight"] = min_child_weight_options[best["min_child_weight"]]

print(best_params)


# In[] Optuna
# https://forecastegy.com/posts/catboost-hyperparameter-tuning-guide-with-optuna/
# https://optuna.readthedocs.io/en/stable/reference/trial.html
# https://github.com/catboost/tutorials/blob/master/hyperparameters_tuning/hyperparameters_tuning_using_optuna_and_hyperopt.ipynb
# https://neptune.ai/blog/optuna-vs-hyperopt



model_name = "XGBoost"


# define models and parameters
def objective(trial):
    params = {
        "n_estimators": 200,
        "learning_rate": trial.suggest_categorical("learning_rate", [0.2, 0.1, 0.01, 0.001]),
        "max_depth": trial.suggest_categorical("max_depth", [3, 4, 5, 6, 7, 8]),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 2, 3, 5, 8, 10]),
    }

    model = xgb.XGBClassifier(objective='binary:logistic', **params)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1985)
    model_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring = "roc_auc").mean()
    return model_auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print('Best hyperparameters:', study.best_params)
print('Best AUC:', study.best_value)


