import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print("=== ANALYSE EXPLORATOIRE DES DONNÉES (EDA) ===")

# 1. Charger DEUX datasets : original pour distributions, encodé pour corrélations
try:
    df_original = pd.read_csv('CLEANED_DATA_FROM_OUTLIERS.csv')
    print("✅ Fichier chargé pour distributions: CLEANED_DATA_FROM_OUTLIERS.csv")
except FileNotFoundError:
    print("❌ Fichier CLEANED_DATA_FROM_OUTLIERS.csv non trouvé!")
    exit()

try:
    df_encoded = pd.read_csv('encoded.csv')
    print("✅ Fichier chargé pour corrélations: encoded.csv")
except FileNotFoundError:
    print("⚠️ Fichier encoded.csv non trouvé, utilisation des données originales pour corrélations")
    df_encoded = df_original.copy()

print(f"Dimensions données originales: {df_original.shape}")
print(f"Dimensions données encodées: {df_encoded.shape}")

# =========================
# 2. DEBUG DES VARIABLES
# =========================
print(f"\n=== DEBUG DES VARIABLES CLÉS ===")

debug_variables = ['Prix', 'Kilométrage', 'Année-Modèle', 'Puissance fiscale', 'État', 'Nombre de portes']

for col in debug_variables:
    if col in df_original.columns:
        print(f"\n📊 {col}:")
        print(f"   Type: {df_original[col].dtype}")
        print(f"   Valeurs uniques: {sorted(df_original[col].unique())}")
        if df_original[col].dtype in ['object', 'category']:
            print(f"   Distribution: {df_original[col].value_counts().to_dict()}")
        else:
            print(f"   Moyenne: {df_original[col].mean():.2f}")
            print(f"   Médiane: {df_original[col].median():.2f}")
            print(f"   Min: {df_original[col].min()}, Max: {df_original[col].max()}")
    else:
        print(f"\n❌ {col}: COLONNE MANQUANTE")
# 3. DISTRIBUTION DES VARIABLES AVEC DONNÉES ORIGINALES
print(f"\n=== DISTRIBUTION DES VARIABLES CLÉS (Données Originales) ===")

variables_analyse = ['Prix', 'Kilométrage', 'Année-Modèle', 'Puissance fiscale', 'État', 'Nombre de portes']
variables_disponibles = [col for col in variables_analyse if col in df_original.columns]

print(f"Variables analysées: {variables_disponibles}")

# Créer les visualisations pour données originales
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, col in enumerate(variables_disponibles[:6]):
    if df_original[col].notna().sum() > 0:
        
        # 🔥 CORRECTION : TRAITEMENT SPÉCIAL POUR "NOMBRE DE PORTES"
        if col == 'Nombre de portes':
            # DIAGRAMME EN BARRES pour variable catégorielle
            portes_counts = df_original['Nombre de portes'].value_counts().sort_index()
            
            bars = axes[i].bar(portes_counts.index.astype(str), portes_counts.values, 
                              color=['lightcoral', 'lightblue', 'lightgreen'])
            axes[i].set_title(f'Distribution de {col}', fontweight='bold')
            axes[i].set_xlabel('Nombre de portes')
            axes[i].set_ylabel('Nombre de véhicules')
            
            # Ajouter les valeurs sur les barres
            for bar, count in zip(bars, portes_counts.values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                            f'{count}', ha='center', va='bottom', fontweight='bold')
                
            # Ajouter les statistiques
            mean_val = df_original[col].mean()
            median_val = df_original[col].median()
            axes[i].text(0.05, 0.95, f'Moyenne: {mean_val:.1f}\nMédiane: {median_val:.1f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        elif col == 'État':
            # Diagramme en barres pour État (déjà corrigé)
            etat_counts = df_original['État'].value_counts()
            etat_order = ['Pour Pièces', 'Endommagé', 'Correct', 'Bon', 'Très bon', 'Excellent', 'Neuf']
            etat_counts = etat_counts.reindex([x for x in etat_order if x in etat_counts.index])
            
            bars = axes[i].bar(etat_counts.index, etat_counts.values, 
                              color=['red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen', 'blue'])
            axes[i].set_title(f'Distribution de {col}', fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Fréquence')
            axes[i].tick_params(axis='x', rotation=45)
            
        elif df_original[col].dtype in ['object', 'category']:
            # Autres variables catégorielles
            value_counts = df_original[col].value_counts().head(10)
            bars = axes[i].bar(value_counts.index.astype(str), value_counts.values, color='skyblue')
            axes[i].set_title(f'Distribution de {col}', fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Fréquence')
            axes[i].tick_params(axis='x', rotation=45)
            
        else:
            # Variables numériques continues (Prix, Kilométrage, etc.)
            axes[i].hist(df_original[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_title(f'Distribution de {col}', fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Fréquence')
            
            # Ajouter des statistiques
            mean_val = df_original[col].mean()
            median_val = df_original[col].median()
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Moyenne: {mean_val:.0f}')
            axes[i].axvline(median_val, color='green', linestyle='--', label=f'Médiane: {median_val:.0f}')
            axes[i].legend()
            
    else:
        axes[i].text(0.5, 0.5, f'Pas de données\npour {col}', 
                    ha='center', va='center', transform=axes[i].transAxes)

# Cacher les axes non utilisés
for i in range(len(variables_disponibles), 6):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig('distribution_variables_originales.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Graphique distribution sauvegardé: distribution_variables_originales.png")
# =========================
# 4. BOXPLOTS DES VARIABLES NUMÉRIQUES
# =========================
print(f"\n=== BOXPLOTS DES VARIABLES NUMÉRIQUES ===")

numeric_vars = ['Prix', 'Kilométrage', 'Année-Modèle', 'Puissance fiscale']
numeric_vars = [col for col in numeric_vars if col in df_original.columns and df_original[col].dtype not in ['object', 'category']]

if numeric_vars:
    n_cols = min(len(numeric_vars), 4)
    fig, axes = plt.subplots(1, n_cols, figsize=(15, 6))
    
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(numeric_vars[:n_cols]):
        sns.boxplot(y=df_original[col], ax=axes[i], color='lightcoral')
        axes[i].set_title(f'Boxplot de {col}')
        axes[i].set_ylabel(col)
    
    plt.tight_layout()
    plt.savefig('boxplots_variables.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Boxplots sauvegardés: boxplots_variables.png")

# =========================
# 5. MATRICE DE CORRÉLATION COMPLÈTE
# =========================
print(f"\n=== MATRICE DE CORRÉLATION COMPLÈTE (Données Encodées) ===")

# Préparer les données encodées
if 'Prix' in df_encoded.columns:
    df_encoded['Prix'] = pd.to_numeric(df_encoded['Prix'], errors='coerce')

# Sélectionner les variables pour la corrélation
variables_correlation = [
    'Prix', 'Kilométrage', 'Année-Modèle', 'Puissance fiscale',
    'État', 'Nombre de portes', 'Première main', 'Boite de vitesses'
]
variables_correlation = [col for col in variables_correlation if col in df_encoded.columns]

if len(variables_correlation) > 1:
    # Calculer la matrice de corrélation
    corr_matrix = df_encoded[variables_correlation].corr()
    
    # Créer le heatmap SANS masque
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap="RdBu_r", 
                fmt=".2f", 
                center=0,
                square=True, 
                cbar_kws={"shrink": .8},
                annot_kws={"size": 10},
                linewidths=0.5,
                linecolor='white')
    
    plt.title("MATRICE DE CORRÉLATION COMPLÈTE", fontsize=16, pad=20, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('matrice_correlation_complete.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Matrice de corrélation COMPLÈTE sauvegardée: matrice_correlation_complete.png")
    
    # =========================
    # 6. CORRÉLATIONS AVEC LE PRIX
    # =========================
    print(f"\n=== CORRÉLATIONS AVEC LE PRIX ===")
    prix_corr = corr_matrix['Prix'].drop('Prix').sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(prix_corr.index, prix_corr.values,
                    color=['green' if x > 0 else 'red' for x in prix_corr.values])
    plt.xlabel('Coefficient de Corrélation', fontweight='bold')
    plt.title('IMPACT DES VARIABLES SUR LE PRIX', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Ajouter les valeurs sur les barres
    for bar, value in zip(bars, prix_corr.values):
        plt.text(bar.get_width() + (0.01 if value >= 0 else -0.03),
                bar.get_y() + bar.get_height()/2,
                f'{value:.3f}',
                ha='left' if value >= 0 else 'right',
                va='center',
                fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('correlation_prix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Graphique corrélation avec prix sauvegardé: correlation_prix.png")
    
    # Afficher le détail des corrélations
    print("\nDÉTAIL DES CORRÉLATIONS:")
    for variable, corr in prix_corr.items():
        interpretation = ""
        if abs(corr) > 0.7: interpretation = "🚀 TRÈS FORTE"
        elif abs(corr) > 0.5: interpretation = "💪 FORTE"
        elif abs(corr) > 0.3: interpretation = "👍 MODÉRÉE"
        elif abs(corr) > 0.1: interpretation = "📊 FAIBLE"
        else: interpretation = "📉 TRÈS FAIBLE"
        
        direction = "⬆️ AUGMENTE" if corr > 0 else "⬇️ DIMINUE"
        print(f"  {variable:.<20} {corr:+.3f} ({interpretation}) {direction} le prix")

# =========================
# 7. ANALYSE DES RELATIONS ENTRE VARIABLES
# =========================
print(f"\n=== ANALYSE DES RELATIONS ENTRE VARIABLES ===")

if len(variables_correlation) > 1:
    print("\nRELATIONS IMPORTANTES ENTRE VARIABLES:")
    relations_importantes = [
        ('Année-Modèle', 'Kilométrage', -0.39, "Voitures récentes ont moins de km"),
        ('Année-Modèle', 'État', 0.37, "Voitures récentes en meilleur état"),
        ('Année-Modèle', 'Première main', 0.39, "Voitures récentes souvent première main"),
        ('Boite de vitesses', 'Puissance fiscale', 0.32, "Boîte auto sur voitures plus puissantes"),
        ('Première main', 'Kilométrage', -0.25, "Premières mains ont moins de km")
    ]
    
    for var1, var2, expected_corr, explication in relations_importantes:
        if var1 in corr_matrix.columns and var2 in corr_matrix.columns:
            corr_reelle = corr_matrix.loc[var1, var2]
            statut = "✅" if abs(corr_reelle - expected_corr) < 0.1 else "⚠️"
            print(f"  {statut} {var1} ~ {var2}: {corr_reelle:.2f} | {explication}")

# =========================
# 8. ANALYSE DES CATÉGORIES
# =========================
print(f"\n=== ANALYSE DES VARIABLES CATÉGORIELLES ===")

categorical_cols = df_original.select_dtypes(include=['object']).columns.tolist()
if len(categorical_cols) > 0 and 'Prix' in df_original.columns:
    for col in categorical_cols[:2]:  # Analyser seulement 2 catégories
        if df_original[col].nunique() <= 10:  # Uniquement si peu de modalités
            print(f"\n📊 {col}:")
            print(f"   Nombre de catégories: {df_original[col].nunique()}")
            
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df_original, x=col, y='Prix')
            plt.title(f'Prix par {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'prix_par_{col}.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✅ Graphique prix par {col} sauvegardé")

# =========================
# 9. RAPPORT FINAL
# =========================
print(f"\n" + "="*60)
print("📈 RAPPORT EDA TERMINÉ AVEC SUCCÈS!")
print("="*60)
print(f"📊 Dimensions données originales: {df_original.shape}")
print(f"📊 Dimensions données encodées: {df_encoded.shape}")
print(f"💰 Variable cible: Prix")

if 'Prix' in df_original.columns:
    print(f"\n📊 STATISTIQUES PRIX (Original):")
    print(f"   • Minimum: {df_original['Prix'].min():,.0f} DH")
    print(f"   • Maximum: {df_original['Prix'].max():,.0f} DH")
    print(f"   • Moyenne: {df_original['Prix'].mean():,.0f} DH")
    print(f"   • Médiane: {df_original['Prix'].median():,.0f} DH")
    print(f"   • Écart-type: {df_original['Prix'].std():,.0f} DH")

print(f"\n📈 GRAPHIQUES GÉNÉRÉS:")
print(f"   ✅ distribution_variables_originales.png")
print(f"   ✅ boxplots_variables.png")
print(f"   ✅ matrice_correlation_complete.png") 
print(f"   ✅ correlation_prix.png")

print(f"\n🎯 VARIABLES LES PLUS IMPORTANTES POUR LE PRIX:")
if 'Prix' in df_encoded.columns and len(variables_correlation) > 1:
    top_variables = corr_matrix['Prix'].abs().sort_values(ascending=False).drop('Prix').head(3)
    for i, (var, corr) in enumerate(top_variables.items(), 1):
        print(f"   {i}. {var} (corrélation: {corr:+.3f})")

print(f"\n🎯 RECOMMANDATIONS POUR LA MODÉLISATION:")
print(f"   • Utiliser Boite de vitesses et Année-Modèle comme features principales")
print(f"   • Inclure État et Première main comme variables secondaires")
print(f"   • Vérifier la colinéarité entre Année-Modèle et Kilométrage")

print(f"\n🚀 PRÊT POUR LA MODÉLISATION MACHINE LEARNING!")