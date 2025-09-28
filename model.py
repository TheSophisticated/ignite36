import streamlit as st
import pandas as pd
import joblib

# Load the model dict
model_data = joblib.load("replenishment_model.pkl")
model = model_data['model']
feature_columns = model_data['feature_columns']

# Load your dataset
df = pd.read_csv("output1.csv")
villages = df['VILLAGE_NA'].unique()

# Village selection
selected_village = st.selectbox("Select Village", villages)
village_df = df[df['VILLAGE_NA'] == selected_village].sort_values('DATE')
latest_row = village_df.iloc[-1]

# Compute features (must match feature_columns exactly)
features_dict = {
    'WL(mbgl)': latest_row['WL(mbgl)'],
    'WL_7day_avg': village_df['WL(mbgl)'].tail(7).mean(),
    'WL_30day_avg': village_df['WL(mbgl)'].tail(30).mean(),
    'Daily_Replenishment': 0,    # optional: compute properly if you want
    'Weekly_Replenishment': 0,
    'Replenish_Trend_7d': 0,
    'Replenish_Trend_30d': 0,
    'Month': pd.Timestamp.today().month,
    'Season_Factor': 4.0,        # optional: compute based on month
    'Depth_Factor': 1.0,
    'Unconfined': 4.0
}

# Make sure order matches training
X_new = pd.DataFrame([features_dict])[feature_columns]

# Predict
prediction = model.predict(X_new)[0]
st.write(f"Predicted replenishment for {selected_village}: {prediction:.3f} meters")
