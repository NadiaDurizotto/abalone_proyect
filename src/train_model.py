from datetime import datetime
import json
from pathlib import Path
import joblib
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


df = pd.read_csv("abalone_proyect/data/processed/abalone_clean.csv")

X = df.drop(columns=["Age", "Rings"])
y = df["Age"]


RANDOM_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)


def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
        random_state=RANDOM_STATE, n_jobs=-1)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    rmse = root_mean_squared_error(y_test, y_pred)
    return rmse


study = optuna.create_study(direction="minimize")

study.optimize(objective, n_trials=20)
best_params = study.best_params


best_n_estimators = best_params["n_estimators"]
best_max_depth = best_params["max_depth"]
best_min_samples_split = best_params["min_samples_split"]

best_rf = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth,
    min_samples_split=best_min_samples_split, random_state=RANDOM_STATE)
best_rf.fit(X_train, y_train)


y_pred_best = best_rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_best)
rmse = root_mean_squared_error(y_test, y_pred_best)
r2 = r2_score(y_test, y_pred_best)


# GUARDAR MODELO
joblib.dump(best_rf, "abalone_proyect/models/model_v2.pkl", compress=3)

# GUARDAR METRICAS
results_dir = Path("abalone_proyect/results")
results_dir.mkdir(parents=True, exist_ok=True)

metrics_data = {
    "random_state": RANDOM_STATE,
    "hyperparameters": best_params,
    "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
}

with open(results_dir / "metrics_v2.json", "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=4)

# GUARDAR RESULTADOS
log_path = results_dir / "experiments_log.csv"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

new_experiment = pd.DataFrame([{
    "experiment_id": "exp_optuna_v2",
    "timestamp": timestamp,
    "model_version": "model_v2",
    "n_estimators": best_n_estimators,
    "max_depth": best_max_depth,
    "random_state": RANDOM_STATE,
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
}])

if log_path.exists():
    new_experiment.to_csv(log_path, mode="a", header=False, index=False)
else:
    new_experiment.to_csv(log_path, mode="w", header=True, index=False)