# Abalone Dataset
Repositorio para el desarrollo del trabajo práctico de Laboratorio de Datos, enfocado en la utilizacion de git en la creacion de proyectos.

## Estructura del Proyecto
abalone_project/ 
│ 
├── data/ 
│   
├── raw/                  
│    
│ 
├── processed/            
└── data_log.json         
├── src/ 
│   
├── preprocess.py         
│   
│ 
└── train_model.py        
├── results/ 
│   
├── metrics_v1.json       
│   
└── experiments_log.csv
│ 
├── models/                   
│ 
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
   Ejecutar `python src/train_model.py` desde la raíz del proyecto. En esta rama, el script optimiza `LinearRegression` con Optuna, comparando `fit_intercept=True` y `False` mediante validación cruzada de 5 particiones. La división train/test es 70/30 y usa `random_state=42`.

   El modelo se guarda comprimido en `models/model_v2_linear_optuna.pkl`, las métricas en `results/metrics_v2_linear_optuna.json` y el experimento se agrega a `results/experiments_log.csv`. El Random Forest `v1` se conserva como referencia.

## Resultados

| Versión | Modelo                    | Hiperparámetros                   | MAE    | RMSE   | R²     |
| v1      | RandomForestRegressor     | `n_estimators=191`, `max_depth=8` | 1.5225 | 2.1690 | 0.5367 |
| v2      | LinearRegression + Optuna | `fit_intercept=True`              | 1.5832 | 2.1874 | 0.5288 |