# import pandas as pd
# import joblib

# # Load your trained model
# model = joblib.load("model/random_forest_model.pkl") 

# # Load encoded training CSV to get the feature list
# encoded_df = pd.read_csv("encoded.csv")  # Adjust path to your file
# model_features = encoded_df.drop(columns=["Prix"], errors='ignore').columns  # Replace 'Prix' if your target column is named differently

# # Raw user input
# input_data = {
#     "Année-Modèle": 2013,
#     "Boite de vitesses": "Manuelle",
#     "Type de carburant": "Diesel",
#     "Kilométrage": 42500.0,
#     "Marque": "Dacia",
#     "Modèle": "Lodgy",
#     "Nombre de portes": 5.0,
#     "Origine": "WW au Maroc",
#     "Première main": "Oui",
#     "Puissance fiscale": "6 CV",
#     "État": "Très bon"
# }


# # Start transforming input
# transformed = {}

# # Basic numerical & label-encoded fields
# transformed["Année-Modèle"] = max(input_data["Année-Modèle"], 1980)
# transformed["Boite de vitesses"] = {"Manuelle": 0, "Automatique": 1}.get(input_data["Boite de vitesses"], 0)
# transformed["Kilométrage"] = input_data["Kilométrage"]
# transformed["Nombre de portes"] = input_data["Nombre de portes"]
# transformed["Première main"] = {"Non": 0, "Oui": 1}.get(input_data["Première main"], 0)

# # Puissance fiscale
# pf = input_data["Puissance fiscale"]
# if pf == "Plus de 41 CV":
#     pf = "42 CV"
# transformed["Puissance fiscale"] = int(pf.replace(" CV", ""))

# # État encoding
# etat_mapping = {
#     "Neuf": 6, "Excellent": 5, "Très bon": 4,
#     "Bon": 3, "Correct": 2, "Endommagé": 1, "Pour Pièces": 0
# }
# transformed["État"] = etat_mapping.get(input_data["État"], 0)

# # Categorical one-hot fields
# categorical_fields = ["Type de carburant", "Origine", "Marque", "Modèle"]

# for feature in model_features:
#     if feature in transformed:
#         continue
#     matched = False
#     for cat in categorical_fields:
#         prefix = f"{cat}_"
#         if feature.startswith(prefix):
#             category_value = feature[len(prefix):]
#             transformed[feature] = int(input_data.get(cat) == category_value)
#             matched = True
#             break
#     if not matched:
#         # Fill all remaining unmatched features with 0
#         transformed[feature] = 0

# # Convert to DataFrame and align columns
# input_df = pd.DataFrame([transformed])
# input_df = input_df[model_features]  # Ensure correct column order

# # Predict
# prediction = model.predict(input_df)[0]
# print(f"✅ Predicted Price: {prediction:.2f} MAD")


import pandas as pd
import joblib

# -------------------- LOAD MODEL --------------------
model = joblib.load(r"model/random_forest_model.pkl")

# Load encoded CSV to get EXACT feature list used during training
encoded_df = pd.read_csv("encoded.csv")
model_features = encoded_df.drop(columns=["Prix"], errors='ignore').columns


# -------------------- RAW USER INPUT --------------------
input_data = {
    "Année-Modèle": 2013,
    "Boite de vitesses": "Manuelle",
    "Type de carburant": "Diesel",
    "Kilométrage": 42500.0,
    "Marque": "Dacia",
    "Modèle": "Lodgy",
    "Nombre de portes": 5.0,
    "Origine": "WW au Maroc",
    "Première main": "Oui",
    "Puissance fiscale": "6 CV",
    "État": "Très bon"
}


# -------------------- TRANSFORMATION --------------------
transformed = {}

# Numeric fields
transformed["Année-Modèle"] = max(input_data["Année-Modèle"], 1980)
transformed["Kilométrage"] = float(input_data["Kilométrage"])
transformed["Nombre de portes"] = float(input_data["Nombre de portes"])

# Boite de vitesses
transformed["Boite de vitesses"] = {
    "Manuelle": 0,
    "Automatique": 1
}.get(input_data["Boite de vitesses"], 0)

# Première main
transformed["Première main"] = {"Non": 0, "Oui": 1}.get(input_data["Première main"], 0)

# Puissance fiscale
pf = input_data["Puissance fiscale"]
if pf == "Plus de 41 CV":
    pf = "42 CV"
transformed["Puissance fiscale"] = int(pf.replace(" CV", ""))

# État
etat_mapping = {
    "Neuf": 6, "Excellent": 5, "Très bon": 4,
    "Bon": 3, "Correct": 2, "Endommagé": 1, "Pour Pièces": 0
}
transformed["État"] = etat_mapping[input_data["État"]]


# -------------------- ONE-HOT ENCODING --------------------
categorical_fields = ["Type de carburant", "Origine", "Marque", "Modèle"]

for col in model_features:
    # Already filled numeric/base columns
    if col in transformed:
        continue

    filled = False

    # Try matching one-hot prefix
    for cat in categorical_fields:
        prefix = f"{cat}_"
        if col.startswith(prefix):
            category_value = col[len(prefix):]  # the exact name after prefix
            transformed[col] = int(input_data[cat] == category_value)
            filled = True
            break

    # If not a known column (rare), fill with 0
    if not filled:
        transformed[col] = 0


# -------------------- ALIGN & PREDICT --------------------
input_df = pd.DataFrame([transformed])
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
  # Align columns perfectly

prediction = model.predict(input_df)[0]
print(f"✅ Predicted Price: {prediction:.2f} MAD") #✅ Predicted Price: 95647.00 MAD
