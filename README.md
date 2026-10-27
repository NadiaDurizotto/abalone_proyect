# Abalone Dataset

Repositorio para el desarrollo del trabajo práctico de Laboratorio de Datos, enfocado en la utilizacion de git en la creacion de proyectos.

## Estructura del Proyecto
```
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
```

## Instrucciones para ejecutar el preprocesamiento

### Preparar el entorno

Desde la raíz del proyecto, en Windows PowerShell:
```
conda create -n abalone python=3.11 pip -y
conda activate abalone
python -m pip install -r requirements.txt
```

### Ejecutar el proyecto

Ejecutá estos comandos desde la raíz del proyecto:
```
python src/preprocess.py
python src/train_model.py
```
El preprocesamiento lee data/raw/abalone_raw.csv y genera data/processed/abalone_clean.csv.

El entrenamiento genera los modelos y las métricas de las variantes incluidas en src/train_model.py. Los modelos se guardan en models/, las métricas en results/ y cada ejecución se agrega a results/experiments_log.csv.

Al repetir el entrenamiento, los archivos de modelos y métricas pueden actualizarse, y se agrega una nueva fila al registro de experimentos.


## Resultados
   - **Experimento exp_optuna_v2 (model_v2):** Optimización de hiperparámetros con Optuna. 
   Métricas obtenidas: R² = [0.53], MAE = [1.52], RMSE = [2.16].
   - **Experimento exp_003 (model_v3):** Optimización de hiperparámetros con Optuna (KNN + RobustScaler). 
   Métricas obtenidas: R² = [0.52], MAE = [1.54], RMSE = [2.20].
   - **Experimento exp_optuna_v4 (model_v4):** Optimización de hiperparámetros con Optuna. 
   Métricas obtenidas: R² = [0.52], MAE = [1.58], RMSE = [2.18].
