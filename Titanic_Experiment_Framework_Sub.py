# Titanic_Experiment_Framework_submission.py

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from data_cleaning_Sub import preprocess_data

RESULTS_FILE = 'data_analysis\\model_comparison_results.csv'

def analyze_error_rates(X_val, y_val, y_pred, group_cols_list, output_prefix):
    df = X_val.copy()
    df['Survived_true'] = y_val.values
    df['Survived_pred'] = y_pred
    records = []
    for cols in group_cols_list:
        g = (
        df
        .groupby(cols)[['Survived_true','Survived_pred']]
        .apply(lambda sub: 1 - (sub['Survived_true'] == sub['Survived_pred']).mean(), 
               include_groups=False)
        .reset_index(name='ErrorRate')
        )
        g['Group'] = "_".join(cols)
        records.append(g)
    all_errors = pd.concat(records, ignore_index=True)
    fname = f"data_analysis\\{output_prefix}_error_rates.csv"
    all_errors.to_csv(fname, index=False)
    print(f"所有分组 Error Rate 已保存到 {fname}")

def get_model(name, params=None, use_gpu=False):
    params = params or {}
    if name == 'rf':
        return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    elif name == 'xgb':
        extra = {}
        if use_gpu:
            extra.update(tree_method='gpu_hist', gpu_id=0)
        return XGBClassifier(
            n_estimators=params.get('n_estimators', 820),
            max_depth=params.get('max_depth', 4),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 1.0),
            colsample_bytree=params.get('colsample_bytree', 1.0),
            eval_metric='logloss',
            random_state=42,
            **extra
        )
    elif name == 'lgb':
        extra = {}
        if use_gpu:
            extra.update(device='gpu', gpu_device_id=0)
        return LGBMClassifier(
            n_estimators=params.get('n_estimators', 820),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            num_leaves=params.get('num_leaves', 64),
            subsample=params.get('subsample', 1.0),
            colsample_bytree=params.get('colsample_bytree', 1.0),
            random_state=42,
            **extra
        )
    elif name == 'cat':
        extra = {}
        if use_gpu:
            extra.update(task_type='GPU', devices='0')
        return CatBoostClassifier(
            iterations=params.get('iterations', 850),
            depth=params.get('depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            verbose=0,
            random_state=42,
            **extra
        )
    elif name == 'lr':
        return LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError("Unsupported model. Choose from: rf, xgb, lgb, cat, lr")

def log_results(model_name, cv_score, test_score):
    row = pd.DataFrame([{
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Model': model_name,
        'CV_Accuracy': round(cv_score, 4),
        'Test_Accuracy': round(test_score, 4)
    }])
    if os.path.exists(RESULTS_FILE):
        row.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
    else:
        row.to_csv(RESULTS_FILE, index=False)
    print(f"结果已记录到 {RESULTS_FILE}")

# Optuna objective functions
def rf_objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 820, 822),
        'max_depth': trial.suggest_int('max_depth', 5, 5),
        'min_samples_split': trial.suggest_int('min_samples_split', 4, 4),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 2),
        'max_features': trial.suggest_categorical('max_features', ['sqrt','log2',0.6,0.6]),
    }
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    return cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1).mean()

def xgb_objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 800, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'subsample': trial.suggest_float('subsample', 0.5, 0.7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.7),
    }
    model = get_model('xgb', params, use_gpu=False)
    return cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1).mean()

def lgb_objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 800, 900),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 32, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    model = get_model('lgb', params, use_gpu=False)
    return cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1).mean()

def cat_objective(trial, X, y):
    params = {
        'iterations': trial.suggest_int('iterations', 600, 900),
        'depth': trial.suggest_int('depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.04, 0.07),
    }
    model = get_model('cat', params, use_gpu=False)
    return cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1).mean()

def main_eval(model_name, use_gpu):
    # split train/test
    df = pd.read_csv('data\\train.csv')
    X, le, scaler = preprocess_data(df, is_train=True)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # tune if needed
    best_params = {}
    if model_name in ['rf','xgb','lgb','cat']:
        print(f"[INFO] Optuna tuning {model_name} ...")
        study = optuna.create_study(direction='maximize')
        if model_name == 'rf':
            study.optimize(lambda t: rf_objective(t, X_train, y_train), n_trials=5)
        elif model_name == 'xgb':
            study.optimize(lambda t: xgb_objective(t, X_train, y_train), n_trials=100)
        elif model_name == 'lgb':
            study.optimize(lambda t: lgb_objective(t, X_train, y_train), n_trials=100)
        elif model_name == 'cat':
            study.optimize(lambda t: cat_objective(t, X_train, y_train), n_trials=100)
        best_params = study.best_params
        print(f"[INFO] Best params for {model_name}: {best_params}")
        joblib.dump(best_params, f'.pkl\\best_params_{model_name}.pkl')

    # train & eval
    model = get_model(model_name, best_params, use_gpu=use_gpu)
    print(f"[INFO] Training {model_name} ...")
    model.fit(X_train, y_train)
    cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1).mean()
    test_score = accuracy_score(y_test, model.predict(X_test))
    print(f"[{model_name}] CV acc: {cv_score:.4f}, Test acc: {test_score:.4f}")
    log_results(model_name, cv_score, test_score)

    # error rates
    analyze_error_rates(
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True),
        model.predict(X_test),
        group_cols_list=[['Pclass','Sex'],['Title'],['FamilyBucket'],['AgeBucket'],['HasCabin']],
        output_prefix=model_name
    )

    # SHAP
    if model_name in ['rf','xgb','lgb','cat']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        plt.figure()
        shap.summary_plot(shap_values, X_train, show=False)
        plt.tight_layout()
        plt.savefig(f'shap\\shap_summary_{model_name}.png')
        plt.close()

    # save
    joblib.dump(model, f'.pkl\\{model_name}.pkl')
    joblib.dump(le,    '.pkl\\label_encoders.pkl')
    joblib.dump(scaler,'.pkl\\scaler.pkl')

    # submit
    test_df = pd.read_csv('data\\test.csv')
    X_submit, _, _ = preprocess_data(test_df, label_encoders=le, scaler=scaler, is_train=False)
    y_submit = model.predict(X_submit)
    pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_submit})\
      .to_csv(f'submission\\submission_{model_name}.csv', index=False)

def main_full(model_name, use_gpu):
    # full train
    df = pd.read_csv('data\\train.csv')
    X, le, scaler = preprocess_data(df, is_train=True)
    y = df['Survived']

    # tune if needed
    best_params = {}
    if model_name in ['rf','xgb','lgb','cat']:
        print(f"[INFO] Optuna tuning {model_name} ...")
        study = optuna.create_study(direction='maximize')
        if model_name == 'rf':
            study.optimize(lambda t: rf_objective(t, X, y), n_trials=5)
        elif model_name == 'xgb':
            study.optimize(lambda t: xgb_objective(t, X, y), n_trials=100)
        elif model_name == 'lgb':
            study.optimize(lambda t: lgb_objective(t, X, y), n_trials=100)
        elif model_name == 'cat':
            study.optimize(lambda t: cat_objective(t, X, y), n_trials=100)
        best_params = study.best_params
        print(f"[INFO] Best params for {model_name}: {best_params}")
        joblib.dump(best_params, f'.pkl\\best_params_{model_name}.pkl')

    model = get_model(model_name, best_params, use_gpu=use_gpu)
    cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1).mean()
    print(f"[FULL {model_name}] 5-fold CV acc: {cv_score:.4f}")

    print(f"[INFO] Training full {model_name} ...")
    model.fit(X, y)
    
    # —— 全量训练后也计算 Error Rates （这里用训练集作“验证”） —— 
    analyze_error_rates(
        X.reset_index(drop=True),
        y.reset_index(drop=True),
        model.predict(X),
        group_cols_list=[['Pclass','Sex'],
                         ['Title'],
                         ['FamilyBucket'],
                         ['AgeBucket'],
                         ['HasCabin']],
        output_prefix=f"{model_name}_full"
    )
    joblib.dump(model, f'.pkl\\{model_name}_full.pkl')

    test_df = pd.read_csv('data\\test.csv')
    X_submit, _, _ = preprocess_data(test_df, label_encoders=le, scaler=scaler, is_train=False)
    y_submit = model.predict(X_submit)
    pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_submit})\
      .to_csv(f'submission\\submission_{model_name}_full.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    type=str, default='rf',
                        help='模型: rf, xgb, lgb, cat, lr')
    parser.add_argument('--use-gpu', action='store_true',
                        help='启用 GPU 训练（xgb/lgb/cat）')
    parser.add_argument('--full',    action='store_true',
                        help='全量训练模式，不拆分验证集')
    args = parser.parse_args()

    if args.full:
        main_full(args.model, use_gpu=args.use_gpu)
    else:
        main_eval(args.model, use_gpu=args.use_gpu)