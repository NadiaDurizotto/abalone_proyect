from datetime import datetime
import json
from pathlib import Path
import joblib
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "abalone_clean.csv"
MODEL_PATH = ROOT / "models" / "model_v2_linear_optuna.pkl"
METRICS_PATH = ROOT / "results" / "metrics_v2_linear_optuna.json"
LOG_PATH = ROOT / "results" / "experiments_log.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.3
CV_FOLDS = 5

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Age", "Rings"])
y = df["Age"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,)

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Regresion Lineal con Optuna
def objective_lr(trial):
    fit_intercept = trial.suggest_categorical(
        "fit_intercept",
        [True, False],
    )
    modelo = LinearRegression(fit_intercept=fit_intercept)

    puntajes = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)

    return -float(puntajes.mean())


sampler = optuna.samplers.GridSampler(
    {"fit_intercept": [True, False]},
    seed=RANDOM_STATE,
)
study_lr = optuna.create_study(direction="minimize", sampler=sampler)
study_lr.optimize(objective_lr, n_trials=2, n_jobs=1)

best_params_lr = study_lr.best_params
modelo_final = LinearRegression(**best_params_lr)
modelo_final.fit(X_train, y_train)

predicciones = modelo_final.predict(X_test)
mae_lr = float(mean_absolute_error(y_test, predicciones))
rmse_lr = float(root_mean_squared_error(y_test, predicciones))
r2_lr = float(r2_score(y_test, predicciones))

timestamp_lr = datetime.now().astimezone().isoformat(timespec="seconds")
experiment_id_lr = datetime.now().strftime("exp_linear_optuna_%Y%m%d_%H%M%S")
model_version_lr = "model_v2_linear_optuna"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(modelo_final, MODEL_PATH, compress=3)

metrics_data_lr = {
    "experiment_id": experiment_id_lr,
    "timestamp": timestamp_lr,
    "model_version": model_version_lr,
    "algorithm": "LinearRegression",
    "target": "Age",
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "cv_folds": CV_FOLDS,
    "optuna_trials": len(study_lr.trials),
    "best_cv_mae": float(study_lr.best_value),
    "hyperparameters": best_params_lr,
    "metrics": {
        "MAE": mae_lr,
        "RMSE": rmse_lr,
        "R2": r2_lr,
    },
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics_data_lr, f, indent=2, ensure_ascii=False)


# RandomForest con Optuna
def objective_rf(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    return root_mean_squared_error(y_test, y_pred)


study_rf = optuna.create_study(direction="minimize")
study_rf.optimize(objective_rf, n_trials=20)
best_params_rf = study_rf.best_params

best_n_estimators = best_params_rf["n_estimators"]
best_max_depth = best_params_rf["max_depth"]
best_min_samples_split = best_params_rf["min_samples_split"]

best_rf = RandomForestRegressor(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    min_samples_split=best_min_samples_split,
    random_state=RANDOM_STATE
)
best_rf.fit(X_train, y_train)

y_pred_best = best_rf.predict(X_test)
mae_rf = float(mean_absolute_error(y_test, y_pred_best))
rmse_rf = float(root_mean_squared_error(y_test, y_pred_best))
r2_rf = float(r2_score(y_test, y_pred_best))

# Guardar modelo de Random Forest
model_path_rf = ROOT / "models" / "model_v2.pkl"
joblib.dump(best_rf, model_path_rf, compress=3)

# Guardar métricas de Random Forest
metrics_path_rf = ROOT / "results" / "metrics_v2.json"
metrics_data_rf = {
    "random_state": RANDOM_STATE,
    "hyperparameters": best_params_rf,
    "metrics": {"MAE": mae_rf, "RMSE": rmse_rf, "R2": r2_rf},
}
with open(metrics_path_rf, "w", encoding="utf-8") as f:
    json.dump(metrics_data_rf, f, indent=4)


log_columns = [
    "experiment_id",
    "timestamp",
    "model_version",
    "algorithm",
    "hyperparameters",
    "random_state",
    "test_size",
    "MAE",
    "RMSE",
    "R2",
]

if LOG_PATH.exists() and LOG_PATH.stat().st_size > 0:
    previous_log = pd.read_csv(LOG_PATH)
    previous_log = previous_log.reindex(columns=log_columns)
else:
    previous_log = pd.DataFrame(columns=log_columns)

timestamp_rf = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

new_experiment_lr = {
    "experiment_id": experiment_id_lr,
    "timestamp": timestamp_lr,
    "model_version": model_version_lr,
    "algorithm": "LinearRegression",
    "hyperparameters": json.dumps(best_params_lr, sort_keys=True),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "MAE": mae_lr,
    "RMSE": rmse_lr,
    "R2": r2_lr,
}

new_experiment_rf = {
    "experiment_id": "exp_optuna_v2",
    "timestamp": timestamp_rf,
    "model_version": "model_v2",
    "algorithm": "RandomForestRegressor",
    "hyperparameters": json.dumps(best_params_rf, sort_keys=True),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "MAE": mae_rf,
    "RMSE": rmse_rf,
    "R2": r2_rf,
}

new_experiments_df = pd.DataFrame([new_experiment_lr, new_experiment_rf], columns=log_columns)
updated_log = pd.concat([previous_log, new_experiments_df], ignore_index=True)
updated_log.to_csv(LOG_PATH, index=False, encoding="utf-8")