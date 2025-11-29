from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load(r"D:\ENSA\AI\projet_fin_module\model\random_forest_model.pkl")    

# Load encoded training CSV to get the feature list
encoded_df = pd.read_csv(r"../encoded.csv")  # Contains one-hot encoded features
model_features = encoded_df.drop(columns=["Prix"], errors='ignore').columns

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json(force=True)
    print("Received input:", input_data)

    transformed = {}

    # Safe conversion for Année-Modèle
    try:
        annee_modele = int(input_data.get("Année-Modèle", 1980))
    except (ValueError, TypeError):
        annee_modele = 1980
    transformed["Année-Modèle"] = max(annee_modele, 1980)

    # Boite de vitesses
    transformed["Boite de vitesses"] = {"Manuelle": 0, "Automatique": 1}.get(input_data.get("Boite de vitesses", ""), 0)

    # Kilométrage
    try:
        transformed["Kilométrage"] = int(input_data.get("Kilométrage", 0))
    except (ValueError, TypeError):
        transformed["Kilométrage"] = 0

    # Nombre de portes
    try:
        transformed["Nombre de portes"] = int(input_data.get("Nombre de portes", 4))
    except (ValueError, TypeError):
        transformed["Nombre de portes"] = 4

    # Première main
    transformed["Première main"] = {"Non": 0, "Oui": 1}.get(input_data.get("Première main", ""), 0)

    # Puissance fiscale
    pf = input_data.get("Puissance fiscale", "4 CV")
    if pf == "Plus de 41 CV":
        pf = "42 CV"
    try:
        transformed["Puissance fiscale"] = int(pf.replace(" CV", ""))
    except ValueError:
        transformed["Puissance fiscale"] = 4  # Default

    # État
    etat_mapping = {
        "Neuf": 6, "Excellent": 5, "Très bon": 4,
        "Bon": 3, "Correct": 2, "Endommagé": 1, "Pour Pièces": 0
    }
    transformed["État"] = etat_mapping.get(input_data.get("État", ""), 0)

    # One-hot encoding for categorical fields
    categorical_fields = ["Type de carburant", "Origine", "Marque", "Modèle"]
    for feature in model_features:
        if feature in transformed:
            continue
        matched = False
        for cat in categorical_fields:
            prefix = f"{cat}_"
            if feature.startswith(prefix):
                category_value = feature[len(prefix):]
                transformed[feature] = int(input_data.get(cat) == category_value)
                matched = True
                break
        if not matched:
            transformed[feature] = 0  # Default for unknown or missing

    # Final dataframe aligned to model features
    input_df = pd.DataFrame([transformed])
    input_df = input_df[model_features]

    # Predict
    try:
        prediction = model.predict(input_df)[0]
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
