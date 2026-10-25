import json
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


df = pd.read_csv("abalone_proyect/data/processed/abalone_clean.csv")

X = df.drop(columns=["Age", "Rings"])
y = df["Age"]


# ENTRENAMIENTO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

n_estimators = 191
max_depth = 8
rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

rf.fit(X_train, y_train)


# METRICAS
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)



# GUARDAR MODELO
joblib.dump(rf, "abalone_proyect/models/model_v1.pkl", compress=3)

# GUARDAR METRICAS
metrics_data = {
    "random_state": 42,
    "hyperparameters": {
        "n_estimators": n_estimators,
        "max_depth": max_depth
    },
    "metrics": {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
}

with open("abalone_proyect/results/metrics_v1.json", "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=4)


# REGISTRAR EN EL LOG DE EXPERIMENTOS
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

new_experiment = pd.DataFrame([{
    "experiment_id": "exp_001",
    "timestamp": timestamp,
    "model_version": "model_v1",
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "random_state": 42,
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
}])

results_dir = Path("abalone_proyect/results")
log_path = results_dir / "experiments_log.csv"
# Si el archivo ya existe, agregamos la fila sin repetir el encabezado; si no, lo creamos
if log_path.exists():
  new_experiment.to_csv(log_path, mode="a", header=False, index=False)
else:
  new_experiment.to_csv(log_path, mode="w", header=True, index=False)