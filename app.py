import streamlit as st
import os
import pandas as pd
import joblib
from sklearn.exceptions import NotFittedError

MODEL_DIR = "models"

model_labels = {
    "blood_sugar": "Blood Sugar",
    "blood_pressure": "Blood Pressure",
    "diabetes": "Diabetes",
    "fever": "Fever",
    "heart": "Heart Disease",
    "readmission": "Readmission"
}

dropdown_options = {
    "gender": ["Male", "Female"],
    "smoking_history": ["never", "No Info", "current", "former", "not current", "ever"],
    "Fever_Severity": ["Normal", "Mild Fever", "High Fever"],
    "Smoking_Status": ["Never", "Current", "Former"],
    "Alcohol_Consumption": ["Yes", "No"],
    "Physical_Activity_Level": ["Low", "Moderate", "High"],
    "Physical_Activity": ["Active", "Sedentary"],
    "Diet_Type": ["Vegan", "Vegetarian", "Non-Vegetarian"],
    "bp_level": ["High", "Low", "Normal"],
    "Education_Level": ["Primary", "Secondary", "Higher"],
    "Employment_Status": ["Unemployed", "Employed", "Retired"],
    "readmitted": ["NO", "<30", ">30"],
    "change": ["Ch", "No"],
    "diabetesMed": ["Yes", "No"]
}

def load_model(model_name):
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def load_sample_input(model_key):
    dataset_file = model_key.replace("blood_sugar", "bloodsugar").replace("blood_pressure", "bp") + ".csv"
    sample_path = os.path.join("datasets", dataset_file)
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path).dropna()
        if not df.empty:
            return df.iloc[0][:-1].to_dict()
    return {}

def display_sample_inputs(sample_inputs):
    st.markdown("### 🧾 Sample Input Values")
    if not sample_inputs:
        st.info("No sample data found.")
    else:
        for key, value in sample_inputs.items():
            st.markdown(f"- **{key}**: `{value}`")

def get_user_input(model_key):
    st.subheader(f"✍️ Enter Input for {model_labels[model_key]}")
    dataset_file = model_key.replace("blood_sugar", "bloodsugar").replace("blood_pressure", "bp") + ".csv"
    sample_path = os.path.join("datasets", dataset_file)
    df = pd.read_csv(sample_path).dropna()

    form_data = {}
    with st.form(f"{model_key}_form"):
        for col in df.columns[:-1]:
            col_clean = col.strip()
            if col_clean in dropdown_options:
                form_data[col_clean] = st.selectbox(col_clean, dropdown_options[col_clean])
            elif df[col].dtype == 'object':
                form_data[col_clean] = st.text_input(col_clean, value="")
            else:
                form_data[col_clean] = st.text_input(col_clean, value="")
        submit = st.form_submit_button("🔍 Predict")

    if submit:
        try:
            for col in form_data:
                if df[col].dtype != 'object' and col not in dropdown_options:
                    form_data[col] = float(form_data[col])
        except ValueError as ve:
            st.error(f"Invalid numeric input: {ve}")
            return None
        return pd.DataFrame([form_data])
    return None

def predict(model, input_df):
    try:
        return model.predict(input_df)[0]
    except NotFittedError:
        st.error("Model not trained.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    return None

def inject_custom_css():
    st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3 {
            color: #2E8BC0;
        }
        .stButton > button {
            background-color: #2E8BC0;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 10px;
            border: none;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #145DA0;
        }
        .stForm > div {
            background-color: #f1f9ff;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #cce4f6;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        }
        .stMarkdown {
            line-height: 1.6;
        }
        .stInfo {
            background-color: #eaf4fc;
        }
        .stSuccess {
            background-color: #d4edda;
        }
        .css-1aumxhk {  /* Sidebar header */
            background-color: #145DA0 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ---------- MAIN FUNCTION ----------
def main():
    st.set_page_config("Healthcare Predictor", layout="centered")
    inject_custom_css()

    st.title("🩺 Welcome to the Healthcare Prediction System")
    st.markdown("""
This intelligent and user-friendly application allows you to **predict a range of health conditions** using pre-trained Machine Learning models.  
It is designed to assist **patients**, **healthcare professionals**, and **researchers** with early detection and health monitoring.

---

### 🔍 **What This App Can Predict**
- 🩸 **Blood Sugar Levels**
- 🩺 **Blood Pressure**
- 🧪 **Diabetes Risk**
- 🌡️ **Fever Condition**
- ❤️ **Heart Disease Risk**
- 🏥 **Hospital Readmission Probability**

Each prediction model is built using carefully preprocessed datasets and fine-tuned algorithms for **accurate and efficient results**.

---

### 💡 **How to Use**
1. Choose a health prediction model from the buttons below.
2. View **sample input values** for guidance.
3. Enter your data into the form.
4. Click **Predict** to see the model's output.

---

""")

    # Track selected model
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    # Model Buttons
    st.markdown("### 🔘 Choose a Prediction Model")
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    for idx, key in enumerate(model_labels):
        if cols[idx % 3].button(model_labels[key]):
            st.session_state.selected_model = key
            st.rerun()

    model_key = st.session_state.selected_model
    if model_key:
        sample_inputs = load_sample_input(model_key)
        display_sample_inputs(sample_inputs)

        model = load_model(model_key)
        if model:
            input_df = get_user_input(model_key)
            if input_df is not None:
                result = predict(model, input_df)
                if result is not None:
                    st.success(f"✅ Prediction: {result}")
        else:
            st.warning("⚠️ Model not found. Please make sure it’s trained and saved in the models folder.")

    # Footer
    st.markdown("""---  
### 👨‍💻 **About the Creator**  
**Developed by:** Saran Kambala  
**Domain:** Artificial Intelligence & Machine Learning (AIML)  
**Tech Stack:** Python · Streamlit · Scikit-learn · Pandas  

_Thanks for using the app!_  

""")

if __name__ == "__main__":
    main()
