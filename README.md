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
└── experiments_log.csv   # (Ejercicio 3) 
│ 
├── models/                   
│ 
├── .gitignore 
├── requirements.txt 
├── carga_dataset.py 
└── README.md 

## Instrucciones para ejecutar el preprocesamiento
Asegurate de tener tu entorno de Conda activo con las librerías necesarias (requirements.txt) e instalado en la raíz del proyecto.

## Como usarlo
1. ### Preprocesamiento
   Ejecutar el archivo preprocess.py. Este toma los datos crudos del csv 'abalone_raw.csv' y crea ;abalone_clean.csv' con los datos ya procesados.