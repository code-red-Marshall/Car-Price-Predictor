# ----------------- sklearn pickle compatibility shim -----------------
# MUST be at the very top (before loading the model).
import streamlit as st

try:
    import sklearn.compose._column_transformer as _ct_mod
    if not hasattr(_ct_mod, "_RemainderColsList"):
        class _RemainderColsList(list):
            """Compatibility shim for older scikit-learn pickles."""
            pass
        _ct_mod._RemainderColsList = _RemainderColsList
except Exception as _e:
    st.write("Compat shim not applied:", _e)
# ---------------------------------------------------------------------

import pandas as pd
import numpy as np
import pickle as pk  # keep pickle available even if joblib import fails
try:
    import joblib
    _USE_JOBLIB = True
except Exception:
    _USE_JOBLIB = False

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sklearn

st.caption(f"scikit-learn running in app: **{sklearn.__version__}**")

# ---- Load model (joblib preferred; graceful errors) ----
@st.cache_resource
def load_model(path: str):
    try:
        if _USE_JOBLIB:
            model = joblib.load(path)
        else:
            with open(path, "rb") as f:
                model = pk.load(f)
        return model
    except Exception as e:
        st.error(
            "Failed to load the model file 'LRModel.pkl'.\n\n"
            "This usually happens when the library versions on this server "
            "don’t match the ones used to train the model. Pin exact versions "
            "in requirements.txt, or re-save the model with joblib.\n\n"
            f"Underlying error: {type(e).__name__}: {e}"
        )
        st.stop()

def _patch_column_transformer_attributes(model):
    """
    Patch older-fitted ColumnTransformers so they work on newer sklearn.
    Safe when remainder='drop' (no passthrough columns), which is what your pipeline uses.
    """
    try:
        def patch_ct(ct: ColumnTransformer):
            # Only patch when attribute is missing; newer versions already have it
            if not hasattr(ct, "_name_to_fitted_passthrough"):
                # If remainder='drop', an empty dict is valid.
                ct._name_to_fitted_passthrough = {}
        # Pipeline
        if isinstance(model, Pipeline):
            for _, step in model.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    patch_ct(step)
        # Direct ColumnTransformer
        if isinstance(model, ColumnTransformer):
            patch_ct(model)
    except Exception as e:
        st.write("ColumnTransformer compat patch failed:", e)

model = load_model("LRModel.pkl")
# apply compat patch after loading
_patch_column_transformer_attributes(model)   # <-- fixed: no leading space

# ---- UI ----
st.image("Car price dekho.png", use_column_width=True)
st.header("Enter Details : ")

cars = pd.read_csv("Cleaned_Car_data.csv")

# Brands
brands = sorted(cars["company"].dropna().unique().tolist())
company = st.selectbox("Select Car Brand", [""] + brands)

# Models filtered by brand
if company:
    models = sorted(
        cars.loc[cars["company"] == company, "name"].dropna().unique().tolist()
    )
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
    s = str(x).replace(",", "").strip()
    return int(s)

if st.button("Predict"):
    # Validate selections
    missing = []
    if not company: missing.append("Brand")
    if not name: missing.append("Model")
    if not fuel: missing.append("Fuel type")

    try:
        year_int = _to_int(year)
        kms_int = _to_int(kms_driven)
    except Exception:
        st.error("Year and Kilometers must be integers (e.g., 2017 and 45000).")
        st.stop()

    if missing:
        st.warning("Please select: " + ", ".join(missing))
        st.stop()
    if year_int is None or kms_int is None:
        st.warning("Please enter valid numbers for Year and Kilometers.")
        st.stop()

    # Build input row
    X = pd.DataFrame(
        [[company, name, year_int, kms_int, fuel]],
        columns=["company", "name", "year", "kms_driven", "fuel_type"],
    )

    try:
        pred = model.predict(X)
        st.success("Predicted Car Price: ₹ {:.2f}".format(float(np.ravel(pred)[0])))
    except Exception as e:
        st.error(
            "Prediction failed. If your model expects preprocessing "
            "(encoders/scalers), ensure those steps are inside the saved Pipeline."
        )
        st.exception(e)
