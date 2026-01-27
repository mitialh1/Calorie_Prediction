#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 19:03:34 2026

@author: hmitial
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
df = pd.read_csv("/Users/hmitial/Calorie_Prediction/USDA.csv")

features = [
    'Protein', 'TotalFat', 'Carbohydrate',
    'Sugar', 'Sodium', 'Cholesterol'
]

X = df[features].fillna(0)
y = df['Calories'].fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "calorie_model.pkl")

print("Model trained and saved!")
