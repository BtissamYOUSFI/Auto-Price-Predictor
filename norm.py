import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
# 1.

df = pd.read_csv('CLEANED_DATA_FROM_OUTLIERS.csv')

scaler = StandardScaler()


# 1. Standardiser les colonnes numériques

minmax = MinMaxScaler()
standard = StandardScaler()

df['Prix'] = standard.fit_transform(df[['Prix']])
df['Année-Modèle'] = minmax.fit_transform(df[['Année-Modèle']])
df['Kilométrage'] = standard.fit_transform(df[['Kilométrage']])
df['Nombre de portes'] = minmax.fit_transform(df[['Nombre de portes']])
df['Puissance fiscale'] = standard.fit_transform(df[['Puissance fiscale']])
df['État'] = minmax.fit_transform(df[['État']])


df.to_csv('normalized.csv',index=False)