import pandas as pd
df = pd.read_csv('data/raw/abalone_raw.csv')


df['Age'] = df['Rings'] + 1.5

df = pd.get_dummies(df, columns=['Sex'], prefix='Sex', dtype=int)

df.to_csv('abalone_proyect/data/processed/abalone_clean.csv', index=False)