# Startup Profitability Predictor

**Streamlit Web Application to Predict Startup Profitability Using Machine Learning**

---

## Overview

The Startup Profitability Predictor is a **user-friendly web application** built using **Streamlit** and **scikit-learn** that predicts whether a startup is likely to be profitable. The app leverages historical startup data and a trained machine learning model to provide real-time predictions based on key business metrics such as funding rounds, revenue, valuation, employees, market share, and industry/region categorization.

This tool is ideal for **entrepreneurs, investors, and analysts** who want to quickly assess the potential success of a startup based on quantifiable parameters.

---

## Features

1. **Interactive Single Prediction**
   - Input startup details via simple numeric fields and dropdown menus.
   - Select only valid categories for `Industry` and `Region`.
   - Get a prediction: ✅ Profitable or ❌ Not Profitable.
   - View probability score of profitability.

2. **Clean and Visually Appealing UI**
   - Custom **background image** for a professional look.
   - **White text** for readability on dark backgrounds.
   - Organized layout for intuitive user experience.

3. **Robust Machine Learning Backend**
   - Trained **Random Forest Classifier** model.
   - Encoders (`LabelEncoder`) handle categorical features (`Industry`, `Region`).
   - Ensures **feature order consistency** to prevent prediction errors.

4. **Extensible**
   - Can easily add batch CSV predictions in future updates.
   - Encoders and model are saved with **pickle** for reusability.


