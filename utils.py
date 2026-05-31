import joblib
import pandas as pd
import numpy as np
import shap
import google.generativeai as genai

# LOAD ALL COMPONENTS
model = joblib.load("./models/churn_model.pkl")
feature_columns = joblib.load("./models/feature_columns.pkl")
threshold = joblib.load("./models/threshold.pkl")

# PREPROCESS INPUT
def preprocess_input(input_df):
    df = input_df.copy()

    # One-hot encoding (same as training)
    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

    # Align columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df

# PREDICTION FUNCTION
def predict_customer(input_df):
    X = preprocess_input(input_df)

    prob = model.predict_proba(X)[:, 1][0]
    pred = int(prob >= threshold)

    return prob, pred

# RISK SEGMENT
def risk_segment(prob):
    if prob >= 0.6:
        return "Very High Risk"
    elif prob >= 0.4:
        return "High Risk"
    elif prob >= threshold:
        return "Medium Risk"
    else:
        return "Low Risk"

# VALUE SEGMENT
def value_segment(premium):
    if premium > 40000:
        return "High Value"
    elif premium > 25000:
        return "Medium Value"
    else:
        return "Low Value"

# PRIORITY SEGMENT
def priority_segment(risk, value):
    if risk == "Very High Risk" and value == "High Value":
        return "Critical"
    elif risk == "Very High Risk":
        return "High Priority"
    elif risk == "High Risk":
        return "Moderate"
    else:
        return "Low"

# ACTION ENGINE
def retention_action(priority):
    if priority == "Critical":
        return "Immediate call + high discount"
    elif priority == "High Priority":
        return "Call customer + personalized offer"
    elif priority == "Moderate":
        return "Send targeted email + follow-up"
    else:
        return "No immediate action"

# SHAP EXPLAINABILITY
explainer = shap.LinearExplainer(model, np.zeros((1, len(feature_columns))))

def get_shap_values(input_df):
    X = preprocess_input(input_df)
    shap_values = explainer.shap_values(X)
    return shap_values, X

# TOP FEATURES
def get_top_features(shap_vals, X_row, top_n=3):
    feature_names = X_row.index
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1]

    top_features = []
    for i in sorted_idx[:top_n]:
        top_features.append((feature_names[i], shap_vals[i]))

    return top_features

# GEMINI EXPLANATION
def generate_explanation(api_key, risk, priority, top_features, model_name="gemini-flash-latest"):
    if not api_key:
        return "No API key provided."

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        # SAFE MODEL HANDLING
        try:
            model_gemini = genai.GenerativeModel(model_name)
        except:
            model_gemini = genai.GenerativeModel("gemini-1.0-pro")

        reasons = "\n".join([f"- {feat}" for feat, _ in top_features])

        prompt = f"""
        Customer Churn Analysis:

        Risk Level: {risk}
        Priority Level: {priority}

        Key Factors:
        {reasons}

        Explain why this customer might churn and suggest actions.
        """

        response = model_gemini.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Gemini Error: {str(e)}"