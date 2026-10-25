import json
from datetime import datetime
from pathlib import Path

import joblib
import optuna
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split


# Rutas relativas a la raíz del repositorio
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "abalone_clean.csv"
MODEL_PATH = ROOT / "models" / "model_v2_linear_optuna.pkl"
METRICS_PATH = ROOT / "results" / "metrics_v2_linear_optuna.json"
LOG_PATH = ROOT / "results" / "experiments_log.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.3
CV_FOLDS = 5

# Cargar datos: Age es el objetivo.
# Rings se excluye porque se utilizó para calcular Age.
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Age", "Rings"])
y = df["Age"]

# Reservar test antes de optimizar: Optuna solo verá X_train e y_train.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

cv = KFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


def objective(trial):
    fit_intercept = trial.suggest_categorical(
        "fit_intercept",
        [True, False],
    )
    modelo = LinearRegression(fit_intercept=fit_intercept)

    puntajes = cross_val_score(
        modelo,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    # sklearn expresa MAE como negativo para permitir maximizar puntajes.
    return -float(puntajes.mean())


# LinearRegression tiene pocos parámetros útiles para explorar.
sampler = optuna.samplers.GridSampler(
    {"fit_intercept": [True, False]},
    seed=RANDOM_STATE,
)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=2, n_jobs=1)

# Entrenar la configuración elegida con todo el conjunto de entrenamiento.
best_params = study.best_params
modelo_final = LinearRegression(**best_params)
modelo_final.fit(X_train, y_train)

# Evaluar una sola vez sobre test, que no se usó para optimizar.
predicciones = modelo_final.predict(X_test)
mae = float(mean_absolute_error(y_test, predicciones))
rmse = float(root_mean_squared_error(y_test, predicciones))
r2 = float(r2_score(y_test, predicciones))


timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
experiment_id = datetime.now().strftime("exp_linear_optuna_%Y%m%d_%H%M%S")
model_version = "model_v2_linear_optuna"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(modelo_final, MODEL_PATH, compress=3)

metrics_data = {
    "experiment_id": experiment_id,
    "timestamp": timestamp,
    "model_version": model_version,
    "algorithm": "LinearRegression",
    "target": "Age",
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "cv_folds": CV_FOLDS,
    "optuna_trials": len(study.trials),
    "best_cv_mae": float(study.best_value),
    "hyperparameters": best_params,
    "metrics": {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    },
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=2, ensure_ascii=False)

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

# Leer y conservar las filas existentes.
if LOG_PATH.exists() and LOG_PATH.stat().st_size > 0:
    previous_log = pd.read_csv(LOG_PATH)

    if {"n_estimators", "max_depth"}.issubset(previous_log.columns):
        preserved_rows = []

        for row in previous_log.to_dict(orient="records"):
            old_hyperparameters = {
                "n_estimators": int(row["n_estimators"]),
                "max_depth": (
                    None
                    if pd.isna(row["max_depth"])
                    else int(row["max_depth"])
                ),
            }

            preserved_rows.append(
                {
                    "experiment_id": row["experiment_id"],
                    "timestamp": row["timestamp"],
                    "model_version": row["model_version"],
                    "algorithm": "RandomForestRegressor",
                    "hyperparameters": json.dumps(
                        old_hyperparameters,
                        sort_keys=True,
                    ),
                    "random_state": int(row["random_state"]),
                    "test_size": TEST_SIZE,
                    "MAE": row["MAE"],
                    "RMSE": row["RMSE"],
                    "R2": row["R2"],
                }
            )

        previous_log = pd.DataFrame(preserved_rows, columns=log_columns)
    else:
        previous_log = previous_log.reindex(columns=log_columns)
else:
    previous_log = pd.DataFrame(columns=log_columns)

new_experiment = {
    "experiment_id": experiment_id,
    "timestamp": timestamp,
    "model_version": model_version,
    "algorithm": "LinearRegression",
    "hyperparameters": json.dumps(best_params, sort_keys=True),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
}

updated_log = pd.concat(
    [previous_log, pd.DataFrame([new_experiment], columns=log_columns)],
    ignore_index=True,
)
updated_log.to_csv(LOG_PATH, index=False, encoding="utf-8")

print(f"Mejor configuración según validación cruzada: {best_params}")
print(f"MAE en test: {mae:.4f}")
print(f"RMSE en test: {rmse:.4f}")
print(f"R² en test: {r2:.4f}")
print(f"Modelo guardado en: {MODEL_PATH}")
print(f"Métricas guardadas en: {METRICS_PATH}")
print(f"Experimentos registrados en: {LOG_PATH}")