import pandas as pd
import numpy as np

# Charger les données
df = pd.read_csv(r'D:\GI_Ensam\Projects\ML\projet_fin_module\data\avito_pfm_merged.csv')

print("=== DÉBUT DU NETTOYAGE ===")
print(f"Shape initial: {df.shape}")

# 1. Supprimer les doublons
df.drop_duplicates(inplace=True)
print(f"Après suppression doublons: {df.shape}")

# 2. Nettoyer et convertir Prix
df = df.dropna(subset=['Prix'])
df['Prix'] = df['Prix'].astype(str).str.replace(' ', '').str.replace(' DH', '').str.replace(' ', '')
df['Prix'] = pd.to_numeric(df['Prix'], errors='coerce')
df = df.dropna(subset=['Prix'])

# 3. CORRECTION CRITIQUE - NETTOYAGE NOMBRE DE PORTES
print("\n=== NETTOYAGE NOMBRE DE PORTES ===")
print("Distribution AVANT nettoyage:")
print(df['Nombre de portes'].value_counts())

# Convertir en numérique
df['Nombre de portes'] = pd.to_numeric(df['Nombre de portes'], errors='coerce')

# 🔥 CORRECTION : PROTÉGER LES 3 PORTES !
# Compter combien de 3 portes nous avons
count_3_portes = (df['Nombre de portes'] == 3.0).sum()
count_5_portes = (df['Nombre de portes'] == 5.0).sum()

print(f"3 portes avant: {count_3_portes}")
print(f"5 portes avant: {count_5_portes}")

# Remplacer les NaN de façon INTELLIGENTE pour préserver la distribution
nan_count = df['Nombre de portes'].isna().sum()
print(f"NaN à remplacer: {nan_count}")

if nan_count > 0:
    # Calculer la proportion réelle 3 portes vs 5 portes
    total_non_nan = count_3_portes + count_5_portes
    if total_non_nan > 0:
        proportion_3_portes = count_3_portes / total_non_nan
        proportion_5_portes = count_5_portes / total_non_nan
        
        print(f"Proportion 3 portes: {proportion_3_portes:.3f}")
        print(f"Proportion 5 portes: {proportion_5_portes:.3f}")
        
        # Remplacer les NaN selon la proportion réelle
        np.random.seed(42)  # Pour la reproductibilité
        nan_indices = df[df['Nombre de portes'].isna()].index
        
        for idx in nan_indices:
            if np.random.random() < proportion_3_portes:
                df.loc[idx, 'Nombre de portes'] = 3.0
            else:
                df.loc[idx, 'Nombre de portes'] = 5.0

# Convertir en entier
df['Nombre de portes'] = df['Nombre de portes'].astype(int)

print("Distribution APRÈS nettoyage:")
print(df['Nombre de portes'].value_counts())

# 4. Nettoyer Kilométrage (reste identique)
df['Kilométrage'] = df['Kilométrage'].str.replace(' ', '').str.replace(' ', '').str.replace(',', '')

def parse_kilometrage(value):
    if isinstance(value, str) and '-' in value:
        min_value, max_value = value.split('-')
        min_value = int(min_value.strip())
        max_value = int(max_value.strip())
        return round((min_value + max_value) / 2)
    elif isinstance(value, str) and value.isdigit():
        return int(value)
    else:
        return np.nan

df['Kilométrage'] = df['Kilométrage'].apply(parse_kilometrage)
df['Kilométrage'] = df['Kilométrage'].fillna(df['Kilométrage'].median())

# 5. Nettoyer les autres colonnes
df['Origine'] = df['Origine'].fillna(df['Origine'].mode()[0])
df['Première main'] = df['Première main'].fillna('Non')

# 6. Supprimer les lignes avec valeurs manquantes
df.dropna(axis=0, inplace=True)

# 7. Supprimer les lignes "Autres" pour Marque et Modèle
df = df.drop(df[(df['Marque'] == 'Autres') & (df['Modèle'] == 'Autres')].index)
df = df.reset_index(drop=True)

# 8. VÉRIFICATION FINALE CRITIQUE
print("\n=== VÉRIFICATION FINALE NOMBRE DE PORTES ===")
final_distribution = df['Nombre de portes'].value_counts()
for portes, count in final_distribution.items():
    pourcentage = (count / len(df)) * 100
    print(f"  {portes} portes: {count} véhicules ({pourcentage:.1f}%)")

# 9. Vérification finale globale
print("\n=== VÉRIFICATION FINALE GLOBALE ===")
print(f"Shape final: {df.shape}")
print("\nValeurs manquantes:")
print(df.isnull().sum())

# 10. Exporter
df.to_csv('CLEANED_DATA.csv', index=False)
print("\n✅ Cleaning terminé avec succès!")
print(f"Fichier sauvegardé: CLEANED_DATA.csv")