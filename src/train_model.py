from datetime import datetime
from pathlib import Path
import json
import joblib
import pandas as pd
import optuna
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score


df = pd.read_csv("abalone_proyect/data/processed/abalone_clean.csv")

X = df.drop(columns=["Age", "Rings"])
y = df["Age"]

# ENTRENAMIENTO Y TESTEO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# BÚSQUEDA DE HIPERPARÁMETROS CON OPTUNA
def objetivo_knn(trial):
    parametros = {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 25, step=2),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_categorical("p", [1, 2])    # tipo de distancia: 1 = Manhattan, 2 = Euclidean
    }

    modelo = Pipeline([("escalador", RobustScaler()), ("regresor", KNeighborsRegressor(**parametros))])

    puntuaciones = cross_val_score(modelo, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
    return puntuaciones.mean()


estudio = optuna.create_study(direction="maximize")
estudio.optimize(objetivo_knn, n_trials=20)

mejores_params = estudio.best_params

knn_pipeline = Pipeline([("scaler", RobustScaler()), ("regressor", KNeighborsRegressor(**mejores_params))])
knn_pipeline.fit(X_train, y_train)


y_pred = knn_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# GUARDAR MODELO
joblib.dump(knn_pipeline, "abalone_proyect/models/model_v3.pkl", compress=3)

# GUARDAR MÉTRICAS
metrics_data = {
    "hyperparameters": mejores_params,
    "metrics": {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
}

with open("abalone_proyect/results/metrics_v3.json", "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=4)


# REGISTRAR EN EL LOG DE EXPERIMENTOS
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

new_experiment = pd.DataFrame([{
    "experiment_id": "exp_003",
    "timestamp": timestamp,
    "model_version": "model_v3",
    "algorithm": "KNN",
    "hyperparameters": mejores_params,  # Guardamos todos los params acá sin romper columnas
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
}])

results_dir = Path("abalone_proyect/results")
log_path = results_dir / "experiments_log.csv"

if log_path.exists():
    new_experiment.to_csv(log_path, mode="a", header=False, index=False)
else:
    new_experiment.to_csv(log_path, mode="w", header=True, index=False)