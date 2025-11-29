import pandas as pd
import numpy as np


df = pd.read_csv(r'data\CLEANED_DATA_FROM_OUTLIERS.csv')
############################# 'Prix' Clean and convert 'Prix' to float64 #############################
print(df['Prix'].head())
print(df['Prix'].dtype)

# df['Prix'] = df['Prix'].str.replace(' ', '', regex=False)  # Remove non-breaking spaces
# df['Prix'] = df['Prix'].str.replace(' DH', '', regex=False)  # Remove ' DH'
df['Prix'] = df['Prix'].astype(float)  # Convert to float64


############################# 'Puissance fiscale' Clean and convert 'Puissance fiscale' to int32 #############################
df['Puissance fiscale'] = df['Puissance fiscale'].replace('Plus de 41 CV', '42 CV')
df['Puissance fiscale'] = df['Puissance fiscale'].str.replace(' CV', '', regex=False).astype(int)


############################# 'Titre' & 'Équipements' droped #############################
df = df.drop(['Titre'],axis=1)
df = df.drop(['Équipements'],axis=1)


############################# 'Première main' binary encoding #############################
df['Première main'] = df['Première main'].map({'Non': 0, 'Oui': 1})
# print(df['Première main'].head())


############################# 'Boite de vitesses' binary encoding #############################
# Check the unique values first
# print(df['Boite de vitesses'].unique())#['Manuelle' 'Automatique']
df['Boite de vitesses'] = df['Boite de vitesses'].map({'Manuelle': 0, 'Automatique': 1})
# print(df['Boite de vitesses'].head(20))



############################# 'Année-Modèle' String to Int #############################
df['Année-Modèle'] = df['Année-Modèle'].replace('1980 ou plus ancien', '1980')
df['Année-Modèle'] = df['Année-Modèle'].astype(int)


############################# 'État' Label endcoding #############################
# Check the unique values first
# print(df['État'].unique())
etat_mapping = {
    'Neuf': 6,
    'Excellent': 5,
    'Très bon': 4,
    'Bon': 3,
    'Correct': 2,
    'Endommagé': 1,
    'Pour Pièces': 0
}

df['État'] = df['État'].map(etat_mapping)
# print(df['État'].head(20))

############################# 'Type de carburant' one hot endcoding #############################

# # One-hot encoding with dtype=int64
# df = pd.get_dummies(df, columns=['Type de carburant'], dtype='int64')

# ############################# 'Origine' one-hot encoding #############################
# df = pd.get_dummies(df, columns=['Origine'], drop_first=False, dtype='int64')

# ############################# 'Marque' one-hot encoding #############################
# df = pd.get_dummies(df, columns=['Marque'], drop_first=False, dtype='int64')

# ############################# 'Modèle' one-hot encoding #############################
# df = pd.get_dummies(df, columns=['Modèle'], drop_first=False, dtype='int64')

# ---------------------- ONE-HOT ENCODING ----------------------
# IMPORTANT : never drop columns → prediction must match training
one_hot_cols = ['Type de carburant', 'Origine', 'Marque', 'Modèle']

df = pd.get_dummies(df, columns=one_hot_cols, drop_first=False, dtype='int64')




print(df.dtypes.unique())

print(df.head(5))

df.to_csv('encoded.csv',index=False)