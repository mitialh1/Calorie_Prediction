#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 19:12:57 2026

@author: hmitial
"""

import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("calorie_model.pkl")

st.title("🍎 Calorie Prediction App")
st.write("Enter nutritional values to predict calories")

# Input fields
protein = st.number_input("Protein (g)", min_value=0.0)
fat = st.number_input("Total Fat (g)", min_value=0.0)
carbs = st.number_input("Carbohydrates (g)", min_value=0.0)
sugar = st.number_input("Sugar (g)", min_value=0.0)
sodium = st.number_input("Sodium (mg)", min_value=0.0)
cholesterol = st.number_input("Cholesterol (mg)", min_value=0.0)

if st.button("Predict Calories"):
    input_data = np.array([[protein, fat, carbs, sugar, sodium, cholesterol]])
    prediction = model.predict(input_data)
    st.success(f"🔥 Predicted Calories: {prediction[0]:.2f}")
