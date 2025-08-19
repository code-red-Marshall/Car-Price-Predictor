import pandas as pd
import numpy as np
import streamlit as st

# ---- Load model (joblib preferred) ----
try:
    import joblib
    _USE_JOBLIB = True
except Exception:
    import pickle as pk
    _USE_JOBLIB = False

def load_model(path: str):
    try:
        if _USE_JOBLIB:
            return joblib.load(path)
        else:
            with open(path, "rb") as f:
                return pk.load(f)
    except Exception as e:
        st.error(
            "Failed to load the model file 'LRModel.pkl'.\n\n"
            "This usually happens when the library versions on this server "
            "don’t match the ones used to train the model. Make sure your "
            "requirements.txt pins the exact versions you trained with.\n\n"
            f"Underlying error: {type(e).__name__}: {e}"
        )
        st.stop()

model = load_model("LRModel.pkl")

# ---- UI ----
st.image("Car price dekho.png", use_column_width=True)
st.header("Enter Details : ")

cars = pd.read_csv("Cleaned_Car_data.csv")

# Brands
brands = sorted(cars["company"].dropna().unique().tolist())
company = st.selectbox("Select Car Brand", [""] + brands)

# Models filtered by brand
if company:
    models = sorted(cars.loc[cars["company"] == company, "name"].dropna().unique().tolist())
else:
    models = sorted(cars["name"].dropna().unique().tolist())
name = st.selectbox("Select Car Model", [""] + models)

# Year & kms
year = st.text_input("Year", placeholder="Enter the manufacturing year (e.g., 2017)")
kms_driven = st.text_input("Kilometers Travelled", placeholder="Enter kilometers (e.g., 45000)")

# Fuel
fuel_types = sorted(cars["fuel_type"].dropna().unique().tolist())
fuel = st.selectbox("Fuel type", [""] + fuel_types)

# ---- Predict ----
def _to_int(x):
    if x is None or str(x).strip() == "":
        return None
    # remove commas/spaces
    x = str(x).replace(",", "").strip()
    return int(x)

if st.button("Predict"):
    # Validate selections
    missing = []
    if not company: missing.append("Brand")
    if not name: missing.append("Model")
    if not fuel: missing.append("Fuel type")

    year_int = None
    kms_int = None
    try:
        year_int = _to_int(year)
    except Exception:
        st.error("Year must be an integer (e.g., 2017).")
        st.stop()
    try:
        kms_int = _to_int(kms_driven)
    except Exception:
        st.error("Kilometers must be an integer (e.g., 45000).")
        st.stop()

    if missing:
        st.warning("Please select: " + ", ".join(missing))
        st.stop()
    if year_int is None or kms_int is None:
        st.warning("Please enter valid numbers for Year and Kilometers.")
        st.stop()

    # Build input row
    input_row = pd.DataFrame(
        [[company, name, year_int, kms_int, fuel]],
        columns=["company", "name", "year", "kms_driven", "fuel_type"],
    )

    try:
        pred = model.predict(input_row)
        st.success("Predicted Car Price: ₹ {:.2f}".format(float(np.ravel(pred)[0])))
    except Exception as e:
        st.error(
            "Prediction failed. This can happen if the model expects preprocessing "
            "steps (encoders/scalers) that aren’t present in the saved pipeline.\n\n"
            f"Underlying error: {type(e).__name__}: {e}"
        )
