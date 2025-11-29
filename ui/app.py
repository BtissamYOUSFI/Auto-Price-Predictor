import streamlit as st
import requests
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# API endpoint
API_URL = 'http://127.0.0.1:5000/predict'  # Match Flask endpoint

# Custom CSS for improved styling
st.markdown("""
    <style>
    .main-title {
        color: #2c3e50;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
    .stSelectbox, .stNumberInput {
        margin-bottom: 20px;
    }
    .prediction-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
    .section-header {
        color: #3498db;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        padding: 10px 0;
        font-size: 16px;
        margin-top: 20px;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #3498db;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('../data/CLEANED_DATA.csv')

# Try to load data and handle potential errors
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Streamlit UI
st.markdown('<h1 class="main-title">🚗 Vehicle Price Predictor</h1>', unsafe_allow_html=True)

# Info box at the top
st.markdown("""
    <div class="info-box">
        Fill in the vehicle details below to get an estimated price based on our machine learning model.
    </div>
""", unsafe_allow_html=True)

# Create three columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<h3 class="section-header">Vehicle Basics</h3>', unsafe_allow_html=True)
    
    # --- Step 1: Select Brand ---
    selected_marque = st.selectbox("Select Brand", sorted(df["Marque"].dropna().unique()), 
                                   help="Choose the vehicle manufacturer")

    # --- Step 2: Filter Models Based on Brand ---
    models_for_brand = sorted(df[df["Marque"] == selected_marque]["Modèle"].dropna().unique())
    selected_modele = st.selectbox("Select Model", models_for_brand, 
                                   help="Choose the specific model of the vehicle")

    # --- Step 3: Filter the dataframe based on Marque + Modèle ---
    filtered_df = df[(df["Marque"] == selected_marque) & (df["Modèle"] == selected_modele)]

    # --- Step 4: Dynamically filter dropdowns ---
    available_years = sorted(filtered_df["Année-Modèle"].dropna().unique())
    available_gears = sorted(filtered_df["Boite de vitesses"].dropna().unique())
    available_fuels = sorted(filtered_df["Type de carburant"].dropna().unique())
    available_puissances = sorted(filtered_df["Puissance fiscale"].dropna().unique())

    # Year and Transmission
    annee_modele = st.selectbox('Model Year', available_years, 
                                help="Select the manufacturing year of the vehicle")

with col2:
    st.markdown('<h3 class="section-header">Technical Specifications</h3>', unsafe_allow_html=True)
    
    # Transmission and Fuel
    boite_vitesses = st.selectbox('Transmission', available_gears, 
                                  help="Select the transmission type")
    
    type_carburant = st.selectbox('Fuel Type', available_fuels, 
                                  help="Choose the type of fuel")
    
    puissance_fiscale = st.selectbox('Fiscal Power (CV)', available_puissances, 
                                     help="Select the vehicle's fiscal power")
    
    nombre_portes = st.number_input('Number of Doors', min_value=2, max_value=5, value=4,
                                    help="Select number of doors")

with col3:
    st.markdown('<h3 class="section-header">Condition & History</h3>', unsafe_allow_html=True)
    
    # Additional Details
    kilometrage = st.number_input('Mileage (km)', min_value=0, 
                                  help="Enter the total kilometers driven")
    
    origine = st.selectbox('Origin', 
        ['WW au Maroc', 'Dédouanée', 'Importée neuve', 'Pas encore dédouanée'],
        help="Select the vehicle's origin status")

    premiere_main = st.selectbox('First Hand', ['Oui', 'Non'], 
                                 help="Is this the first owner of the vehicle?")

    etat = st.selectbox('Condition', 
        ['Neuf', 'Excellent', 'Très bon', 'Bon', 'Correct', 'Pour Pièces'],
        help="Select the current condition of the vehicle")

# --- Submit button ---
if st.button('Predict Vehicle Price 🔍'):
    # Show spinner while processing
    with st.spinner('Calculating price estimate...'):
        data = {
            'Année-Modèle': annee_modele,
            'Boite de vitesses': boite_vitesses,
            'Type de carburant': type_carburant,
            'Kilométrage': kilometrage,
            'Marque': selected_marque,
            'Modèle': selected_modele,
            'Nombre de portes': nombre_portes,
            'Origine': origine,
            'Première main': premiere_main,
            'Puissance fiscale': puissance_fiscale,
            'État': etat
        }

        try:
            response = requests.post(API_URL, json=data)
            response.raise_for_status()
            prediction = response.json()
            
            # Styled prediction display
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Estimated Price 💰</h3>
                <h2 style="color: #27ae60; font-size: 2.5rem;">{prediction['prediction']:,.2f} DH</h2>
                <p>Based on market data and vehicle specifications</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display vehicle summary
            st.markdown("### Vehicle Summary")
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.write(f"**Brand:** {selected_marque}")
                st.write(f"**Model:** {selected_modele}")
                st.write(f"**Year:** {annee_modele}")
                st.write(f"**Mileage:** {kilometrage} km")
                st.write(f"**Condition:** {etat}")
            
            with summary_col2:
                st.write(f"**Transmission:** {boite_vitesses}")
                st.write(f"**Fuel Type:** {type_carburant}")
                st.write(f"**Power:** {puissance_fiscale} CV")
                st.write(f"**First Hand:** {premiere_main}")
                st.write(f"**Origin:** {origine}")
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error contacting API: {e}")
            st.info("Make sure the prediction API is running at " + API_URL)
        except Exception as ex:
            st.error(f"❌ Unexpected error: {ex}")

# Add expander for additional information
with st.expander("About This Tool"):
    st.write("""
        This vehicle price prediction tool uses machine learning to estimate the market value of a vehicle based on its specifications and condition.
        
        **How it works:**
        1. Fill in all the vehicle details
        2. Click the "Predict Vehicle Price" button
        3. Our AI model will analyze the data and provide an estimated price
        
        Please note that this is an estimate and actual market prices may vary.
    """)

# Add a footer
st.markdown("""
    <div style='text-align: center; color: #7f8c8d; margin-top: 30px; padding: 20px;'>
    🚗 Vehicle Price Predictor | Powered by Machine Learning
    </div>
""", unsafe_allow_html=True)