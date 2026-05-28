import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Page config
st.set_page_config("Random Forest Regression with insurance dataset", layout="centered")


# Load CSS
def load_css(file):
    file_path = os.path.join(SCRIPT_DIR, file)
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("style.css")

# Title
st.markdown(
    """
<div class="card">
<h1>Random Forest Regression with insurance dataset</h1>
<p>Predict <b>Charges</b> from <b>Age, Sex, BMI, Children</b> and other factors</p>
</div>
""",
    unsafe_allow_html=True,
)


# Load data
@st.cache_data
def load_data():
    data_path = os.path.join(SCRIPT_DIR, "data", "insurance.csv")
    return pd.read_csv(data_path)


df = load_data()

# Standardize column names to lowercase
df.columns = df.columns.str.lower()

# Dataset preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown("</div>", unsafe_allow_html=True)

# Load saved artifacts
model_dir = os.path.join(SCRIPT_DIR, "models")
scaler_path = os.path.join(model_dir, "scaler.pkl")
model_path = os.path.join(model_dir, "rf_regressor.pkl")
encoders_path = os.path.join(model_dir, "label_encoders.pkl")

try:
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)
    models_loaded = True
except FileNotFoundError as e:
    st.error(f"❌ Model files not found: {e}")
    st.info("Please train the models first by running the notebook: `notebooks/insurance_eda_and_model.ipynb`")
    st.stop()

# Encode categorical variables with saved encoders
encoded_df = df.copy()
for col in ['sex', 'smoker', 'region']:
    encoded_df[col] = label_encoders[col].transform(encoded_df[col].astype(str).str.lower().str.strip())

# Prepare data
X = encoded_df.drop(columns=["charges"])
y = encoded_df["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

y_pred = model.predict(X_test)

# Metrics
mae=mean_absolute_error(y_test, y_pred)
rmse=np.sqrt(mean_squared_error(y_test, y_pred))
r2=r2_score(y_test, y_pred)
adj_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-X.shape[1]-1)

# Visualization (Charges vs Age)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Insurance Charges vs Age")

fig, ax = plt.subplots()
ax.scatter(df["age"], df["charges"], alpha=0.6)

age_vals = np.linspace(df["age"].min(), df["age"].max(), 100)

# Create feature matrix with average values for other features
mean_values = encoded_df.drop(columns=["charges"]).mean()
X_line = pd.DataFrame([mean_values] * len(age_vals))
X_line["age"] = age_vals

X_line_scaled = scaler.transform(X_line)
y_line = model.predict(X_line_scaled)

ax.plot(age_vals, y_line, color="red", linewidth=2, label="Trend")
ax.set_xlabel("Age")
ax.set_ylabel("Charges")
ax.legend()

st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

# Performance
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R²", f"{r2:.3f}")
c4.metric("Adjusted R²", f"{adj_r2:.3f}")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
<div class="card">
<h3>Model Details</h3>
<p>
<b>Model type:</b> Random Forest Regression<br>
<b>Number of trees:</b> {model.n_estimators}<br>
<b>Max depth:</b> {model.max_depth}
</p>
</div>
""",
    unsafe_allow_html=True,
)

# Prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Charges")

sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", float(df["age"].min()), float(df["age"].max()), 30.0)
bmi = st.slider("BMI", float(df["bmi"].min()), float(df["bmi"].max()), 25.0)
children = st.slider("Children", int(df["children"].min()), int(df["children"].max()), 0)
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Encode inputs using saved encoders
sex_encoded = label_encoders["sex"].transform([sex])[0]
smoker_encoded = label_encoders["smoker"].transform([smoker])[0]
region_encoded = label_encoders["region"].transform([region])[0]

# Create input array in correct feature order
input_data = [[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]]
input_scaled = scaler.transform(input_data)
pred_charges = model.predict(input_scaled)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Charges: ${pred_charges:.2f}</div>',
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)