import streamlit as st
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Crop Recommendation", page_icon="??", layout="centered")

st.title("?? Crop Recommendation")
st.caption("Uses your RandomForest + MinMaxScaler + StandardScaler exactly like your Flask app.")

# ---------- file paths ----------
MODEL_PATH = Path("model/model.pkl")
STAND_PATH = Path("model/standscaler.pkl")     # StandardScaler
MINMAX_PATH = Path("model/minmaxscaler.pkl")   # MinMaxScaler

# ---------- load artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH.open("rb"))
    sc = joblib.load(STAND_PATH.open("rb"))
    ms = joblib.load(MINMAX_PATH.open("rb"))
    return model, sc, ms

try:
    model, sc, ms = load_artifacts()
except Exception as e:
    st.error(f"Failed to load artifacts. Make sure model files exist under /model. Error: {e}")
    st.stop()

# ---------- the same class mapping you used ----------
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

st.subheader("Enter Soil & Weather Inputs")

col1, col2 = st.columns(2)
with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=300.0, value=90.0, step=1.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=300.0, value=42.0, step=1.0)
    K = st.number_input("Potassium (K)", min_value=0.0, max_value=300.0, value=43.0, step=1.0)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
with col2:
    temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=24.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, value=100.0, step=0.1)

# order must match your training / Flask code:
# [N, P, K, temp, humidity, ph, rainfall]
features = np.array([[N, P, K, temp, humidity, ph, rainfall]], dtype=float)

if st.button("Recommend Crop"):
    try:
        scaled_features = ms.transform(features)   # MinMax first
        final_features = sc.transform(scaled_features)  # then StandardScaler
        pred = model.predict(final_features)

        y = int(pred[0])
        crop_name = crop_dict.get(y)
        if crop_name:
            st.success(f"**{crop_name}** is the best crop to be cultivated right there.")
        else:
            st.warning("Sorry, could not map the predicted class to a crop name.")

        # optional: probabilities if available
        try:
            proba = model.predict_proba(final_features)[0]
            # find the predicted class index in model.classes_
            if hasattr(model, "classes_"):
                # classes_ might be [1..22]; map index of predicted class to its probability
                class_index = list(model.classes_).index(y)
                st.write(f"Model confidence: {proba[class_index]:.2%}")
        except Exception:
            pass

        st.caption("Inputs used (in model order): N, P, K, temperature, humidity, pH, rainfall.")
        st.write({ "N": N, "P": P, "K": K, "temperature": temp, "humidity": humidity, "ph": ph, "rainfall": rainfall })

    except Exception as e:
        st.error(f"Prediction failed: {e}")
