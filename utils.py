from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, root_mean_squared_error, mean_absolute_error, r2_score, precision_recall_curve
import numpy as np

drop_cols_leaky = [
    "finishing_position", "laps_completed_pct", "finishing_status", "laps_led", "times_led",
    "points_earned", "diff_laps", "diff_time", "playoff_points_earned", 
    "points_position", "stage_1_position", "stage_2_position", "stage_3_position", 
    "stage_1_stage_points", "stage_2_stage_points", "stage_3_stage_points",
    "mid_ps", "closing_ps", "closing_laps_diff", "avg_ps", "passes_gf", "passing_diff", "passed_gf", 
    "quality_passes", "fast_laps", "top15_laps", "rating", "dnf", "crash",
    "top20", "top10", "top5", "win", "stage_win", "got_stage_points"
]

def chronological_split(df, target_col, test_size):
 
    # splits features and labels chronologically based on race_date (default 80/20 split) 
    df = df.sort_values("race_date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))

    # target_col will always be in drop_cols_leaky along with other in-race attributes
    X = df.drop(columns=drop_cols_leaky + ["race_date"])
    y = df[target_col]

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category")

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def evaluate_classification_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # --- Default threshold (0.5) ---
    y_pred_default = (y_pred_proba >= 0.5).astype(int)

        # --- Find best threshold for F1 ---
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred_best = (y_pred_proba >= best_threshold).astype(int)

    return {
        "AUC": roc_auc_score(y_test, y_pred_proba),
        "Accuracy_default": accuracy_score(y_test, y_pred_default),
        "F1_default": f1_score(y_test, y_pred_default),
        "Accuracy_best": accuracy_score(y_test, y_pred_best),
        "F1_best": f1_score(y_test, y_pred_best),
        "Best_threshold": best_threshold
    }

def evaluate_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Remove NaN values from y_test and corresponding predictions
    mask = ~y_test.isna()
    y_test = y_test[mask]
    y_pred = y_pred[mask]

    return {
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }