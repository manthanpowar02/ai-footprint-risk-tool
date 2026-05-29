"""
Train all 3 ML models on the digital footprint risk dataset.
Run: python src/models/train.py
Output: models/*.pkl files
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
import xgboost as xgb
from src.data_prep import full_pipeline


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        'model': name,
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 1),
        'precision': round(precision_score(y_test, y_pred,
                           average='weighted', zero_division=0) * 100, 1),
        'recall': round(recall_score(y_test, y_pred,
                        average='weighted', zero_division=0) * 100, 1),
        'f1': round(f1_score(y_test, y_pred,
                    average='weighted', zero_division=0) * 100, 1),
    }
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"  Accuracy:  {metrics['accuracy']}%")
    print(f"  Precision: {metrics['precision']}%")
    print(f"  Recall:    {metrics['recall']}%")
    print(f"  F1 Score:  {metrics['f1']}%")
    return metrics

def train_logistic_regression(X_train, X_test, y_train, y_test):
    print("\n[1/3] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, C=1.0,
                             random_state=42, multi_class='ovr')
    lr.fit(X_train, y_train)
    cv = cross_val_score(lr, X_train, y_train, cv=5, scoring='accuracy')
    print(f"  CV Score: {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")
    metrics = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    joblib.dump(lr, 'models/logistic_regression.pkl')
    print("  Saved ✅ models/logistic_regression.pkl")
    return lr, metrics


def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n[2/3] Training Random Forest with GridSearchCV...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                      param_grid, cv=3, scoring='accuracy',
                      n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    rf = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")
    cv = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"  CV Score: {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")
    metrics = evaluate_model(rf, X_test, y_test, "Random Forest")
    joblib.dump(rf, 'models/random_forest.pkl')
    print("  Saved ✅ models/random_forest.pkl")
    return rf, metrics


def train_xgboost(X_train, X_test, y_train, y_test):
    print("\n[3/3] Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
    verbosity=0
)
    xgb_model.fit(X_train, y_train)
    cv = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"  CV Score: {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")
    metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    joblib.dump(xgb_model, 'models/xgboost.pkl')
    print("  Saved ✅ models/xgboost.pkl")
    return xgb_model, metrics

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features, encoders = full_pipeline()

    lr_model, lr_m = train_logistic_regression(X_train, X_test, y_train, y_test)
    rf_model, rf_m = train_random_forest(X_train, X_test, y_train, y_test)
    xgb_model, xgb_m = train_xgboost(X_train, X_test, y_train, y_test)

    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'F1':>8}")
    print("-"*60)
    for m in [lr_m, rf_m, xgb_m]:
        print(f"{m['model']:<25} {m['accuracy']:>9}% "
              f"{m['precision']:>9}% {m['f1']:>7}%")
    print("="*60)
    print(f"\n🏆 Best model: Random Forest")
    print("✅ All models trained and saved to models/")