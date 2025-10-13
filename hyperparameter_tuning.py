import optuna
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

# --- Step 1: Precompute feature importance ranking ---
# You already trained one model, so reuse its feature importance
importances = model.feature_importances_
feature_ranking = np.argsort(importances)[::-1]  # indices sorted by importance, high -> low

# --- Step 2: Define a helper to reduce features ---
def reduce_features(X, keep_ratio):
    """
    Keep top X% features by importance.
    """
    n_features = int(len(feature_ranking) * keep_ratio)
    selected_idx = feature_ranking[:n_features]
    return X.iloc[:, selected_idx]


# --- Step 3: Define Optuna objective ---
def objective(trial):
    # Suggest feature keep ratio
    keep_ratio = trial.suggest_float("keep_ratio", 0.1, 1.0)  # between 10% and 100%
    
    # Reduce training set
    X_train_reduced = reduce_features(X_train, keep_ratio)
    
    # Suggest XGBoost hyperparameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": 42
    }
    
    # Train + cross-validate
    model = XGBRegressor(**params)
    score = cross_val_score(
        model, X_train_reduced, y_train,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1
    ).mean()
    
    return -score  # minimize RMSE


# --- Step 4: Run study ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best trial:")
print("  Params:", study.best_trial.params)
print("  RMSE:", study.best_value)