# EXÉCUTEZ CE CODE D'URGENCE :
import pandas as pd

# Charger les données ORIGINALES (avant tout nettoyage)
df_original = pd.read_csv(r'D:\GI_Ensam\Projects\ML\projet_fin_module\data\avito_pfm_merged.csv')

print("=== INVESTIGATION NOMBRE DE PORTES ===")
print("Valeurs DANS LES DONNÉES BRUTES:")
print(df_original['Nombre de portes'].value_counts())

print("\nTop 10 des valeurs:")
print(df_original['Nombre de portes'].value_counts().head(10))

print("\nType de données:")
print(df_original['Nombre de portes'].dtype)