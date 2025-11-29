import pandas as pd
import numpy as np

print("=== DÉTECTION ET SUPPRESSION DES OUTLIERS ===")

# Charger le DataFrame
df = pd.read_csv("CLEANED_DATA.csv")

# VÉRIFICATION CRITIQUE AVANT TRAITEMENT
print("\n=== VÉRIFICATION NOMBRE DE PORTES AVANT OUTLIERS ===")
print("Valeurs uniques:", sorted(df['Nombre de portes'].unique()))
portes_distribution = df['Nombre de portes'].value_counts().sort_index()
for portes, count in portes_distribution.items():
    pourcentage = (count / len(df)) * 100
    print(f"  {portes} portes: {count} véhicules ({pourcentage:.1f}%)")

# Nettoyer la colonne Prix
df['Prix'] = (
    df['Prix']
    .astype(str)
    .str.replace(' ', '', regex=False)
    .str.replace(' DH', '', regex=False) 
    .str.replace(' ', '')
    .replace('', pd.NA)
    .astype(float)
)

# 🔥 CORRECTION CRITIQUE : NE PAS INCLURE "Nombre de portes" dans les outliers
# C'est une variable CATÉGORIELLE, pas une variable numérique continue !
numeric_columns = ["Prix", "Année-Modèle", "Kilométrage"]  # ⚠️ RETIRER "Nombre de portes"

print(f"\nColonnes analysées pour outliers: {numeric_columns}")

# Convertir uniquement les colonnes numériques
for col in numeric_columns:
    if col == "Année-Modèle":
        df[col] = pd.to_numeric(df[col], errors='coerce').astype("Int64")
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Détection des outliers (UNIQUEMENT sur vraies variables numériques)
outliers_info = {}
mask_global = pd.Series([False] * len(df))

for col in numeric_columns:
    if col in df.columns and df[col].notna().any():
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Protection contre les IQR trop petits
        if IQR == 0:
            # Si pas de variance, utiliser percentiles
            lower_bound = df[col].quantile(0.01)  # 1er percentile
            upper_bound = df[col].quantile(0.99)  # 99ème percentile
        else:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        
        col_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        mask_global = mask_global | col_mask
        
        outliers_count = col_mask.sum()
        outliers_info[col] = {
            'count': outliers_count,
            'percentage': (outliers_count / len(df)) * 100,
            'bounds': (lower_bound, upper_bound)
        }
        
        print(f"Outliers {col}: {outliers_count} ({outliers_info[col]['percentage']:.1f}%)")

# PROTECTION DES CATÉGORIES RARES
print("\n=== PROTECTION DES CATÉGORIES RARES ===")

# Identifier les catégories rares à protéger
categories_rares = {}
for portes in df['Nombre de portes'].unique():
    count = (df['Nombre de portes'] == portes).sum()
    total = len(df)
    pourcentage = (count / total) * 100
    
    # Considérer comme rare si moins de 5% du dataset
    if pourcentage < 5.0:
        categories_rares[portes] = count
        print(f"⚠️  Catégorie rare: {portes} portes - {count} véhicules ({pourcentage:.1f}%)")

# Créer un masque de protection pour les catégories rares
mask_protection = pd.Series([False] * len(df))
if categories_rares:
    for portes_rare in categories_rares.keys():
        indices_categorie_rare = df[df['Nombre de portes'] == portes_rare].index
        mask_protection[indices_categorie_rare] = True
    print(f"Protection appliquée pour {len(categories_rares)} catégorie(s) rare(s)")

# Masque final : outliers MAIS on protège les catégories rares
mask_final = mask_global & (~mask_protection)

total_outliers_final = mask_final.sum()
print(f"\n=== RÉSUMÉ OUTLIERS ===")
print(f"Lignes avec outliers (sans protection): {mask_global.sum()}")
print(f"Lignes avec outliers (avec protection): {total_outliers_final}")
print(f"Pourcentage à supprimer: {total_outliers_final/len(df)*100:.1f}%")

# Filtrer les lignes sans outliers (en protégeant les catégories rares)
df_cleaned = df[~mask_final].copy()

# VÉRIFICATION CRITIQUE APRÈS TRAITEMENT
print("\n=== VÉRIFICATION NOMBRE DE PORTES APRÈS OUTLIERS ===")
portes_apres = df_cleaned['Nombre de portes'].value_counts().sort_index()
for portes, count in portes_apres.items():
    pourcentage_apres = (count / len(df_cleaned)) * 100
    count_avant = portes_distribution.get(portes, 0)
    pourcentage_avant = (count_avant / len(df)) * 100 if count_avant > 0 else 0
    
    print(f"  {portes} portes: {count} véhicules ({pourcentage_apres:.1f}%) [avant: {count_avant} véhicules, {pourcentage_avant:.1f}%]")

# Vérifier la conservation des catégories
portes_avant_set = set(df['Nombre de portes'].unique())
portes_apres_set = set(df_cleaned['Nombre de portes'].unique())
portes_perdues = portes_avant_set - portes_apres_set

if portes_perdues:
    print(f"🚨 ALERTE: Catégories perdues: {portes_perdues}")
    print("Application d'un correctif d'urgence...")
    
    # Correctif : garder au moins 10 échantillons de chaque catégorie perdue
    for portes_perdu in portes_perdues:
        echantillons_a_garder = df[df['Nombre de portes'] == portes_perdu].head(10)
        df_cleaned = pd.concat([df_cleaned, echantillons_a_garder])
        print(f"  + Ajouté {len(echantillons_a_garder)} échantillons de {portes_perdu} portes")
    
    # Réindexer après correction
    df_cleaned = df_cleaned.reset_index(drop=True)

# Statistiques finales
print(f"\n=== COMPARAISON FINALE ===")
print(f"Taille dataset: {len(df)} → {len(df_cleaned)}")
print(f"Taux de conservation: {len(df_cleaned)/len(df)*100:.1f}%")

print("\nDistribution finale du nombre de portes:")
portes_finale = df_cleaned['Nombre de portes'].value_counts().sort_index()
for portes, count in portes_finale.items():
    pourcentage = (count / len(df_cleaned)) * 100
    print(f"  {portes} portes: {count} véhicules ({pourcentage:.1f}%)")

# Sauvegarder le DataFrame nettoyé
df_cleaned.to_csv("CLEANED_DATA_FROM_OUTLIERS.csv", index=False)
print(f"\n✅ Fichier nettoyé enregistré avec succès: CLEANED_DATA_FROM_OUTLIERS.csv")
print("✅ Distribution du nombre de portes PRÉSERVÉE!")