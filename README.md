# Abalone Dataset
Repositorio para el desarrollo del trabajo práctico de Laboratorio de Datos, enfocado en la utilizacion de git en la creacion de proyectos.


## Estructura del Proyecto
abalone_project/
├── data/
│   ├── raw/
│   └── processed/
│   └── data_log.json
├── src/
│   ├── preprocess.py
│   └── train_model.py
├── results/
│   ├── metrics_v1.json
│   └── experiments_log.csv
├── models/
├── .gitignore
├── requirements.txt
├── carga_dataset.py
└── README.md


## Instrucciones para ejecutar el preprocesamiento
Asegurate de tener tu entorno de Conda activo con las librerías necesarias (requirements.txt) e instalado en la raíz del proyecto.


## Como usarlo:
1. ### Preprocesamiento
   Ejecutar el archivo 'preprocess.py'. Este toma los datos crudos del csv 'abalone_raw.csv' y crea 'abalone_clean.csv' con los datos ya procesados.
2. ### Modelado y Entrenamiento
   Ejecutar `python src/train_model.py` desde la raíz del proyecto. El script entrena los modelos y permite realizar la optimización de hiperparámetros mediante Optuna.
   - El modelo base (Random Forest) genera el modelo comprimido `models/model_v1.pkl`, las métricas en `results/metrics_v1.json` y el registro en `results/experiments_log.csv`.
   - La variante con KNN optimizada con Optuna genera `models/model_v3.pkl`, las métricas en 'metrics_v3.json' y el registro en `results/experiments_log.csv`.
   - La variante con Regresión Lineal optimizada con Optuna genera `models/model_v2_linear_optuna.pkl`, las métricas en 'metrics_v2_linear_optuna.json' y el registro en `results/experiments_log.csv`.


## Resultados
   - **Experimento exp_optuna_v2 (model_v2):** Optimización de hiperparámetros con Optuna. 
   Métricas obtenidas: R² = [0.53], MAE = [1.52], RMSE = [2.16].
   - **Experimento exp_003 (model_v3):** Optimización de hiperparámetros con Optuna (KNN + RobustScaler). 
   Métricas obtenidas: R² = [0.52], MAE = [1.54], RMSE = [2.20].
   - **Experimento exp_optuna_v4 (model_v4):** Optimización de hiperparámetros con Optuna. 
   Métricas obtenidas: R² = [0.52], MAE = [1.58], RMSE = [2.18].
