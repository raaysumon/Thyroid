import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# --- Page Config ---
st.set_page_config(
    page_title="Thyroid AI Predictor",
    page_icon="🩺",
    layout="wide"
)

# --- Custom CSS for a Professional Look ---
st.markdown("""
    <style>
    /* Main background color */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Styling the prediction button */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #0056b3;
        border: none;
        color: white;
    }

    /* Card-like styling for sections */
    div[data-testid="stVerticalBlock"] > div:has(div.stColumn) {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)


# --- Model Loading ---
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    try:
        model.load_model("catboost_thyroid_model.cbm")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model = load_model()

# --- Header Section ---
st.title("🩺 Thyroid Cancer Recurrence Analytics")
st.markdown("""
This clinical decision support tool uses a machine learning model (CatBoost) to predict the likelihood of 
thyroid cancer recurrence based on patient pathology and treatment response.
""")
st.divider()

# --- Input Form ---
# We wrap the inputs in a container for better alignment
with st.container():
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.subheader("📋 Patient History")
        hx_rad = st.selectbox("Hx Radiotherapy", ['No', 'Yes'])
        thyroid_func = st.selectbox(
            "Thyroid Function",
            ['Euthyroid', 'Clinical Hyperthyroidism', 'Clinical Hypothyroidism',
             'Subclinical Hyperthyroidism', 'Subclinical Hypothyroidism']
        )

    with col2:
        st.subheader("🔍 Clinical Findings")
        phys_exam = st.selectbox(
            "Physical Examination",
            ['Single nodular goiter-left', 'Multinodular goiter',
             'Single nodular goiter-right', 'Normal', 'Diffuse goiter']
        )
        stage = st.selectbox("TNM Stage", ['I', 'II', 'III', 'IVA', 'IVB'])

    with col3:
        st.subheader("⚡ Risk & Response")
        risk = st.selectbox("Risk Level", ['Low', 'Intermediate', 'High'])
        response = st.selectbox(
            "Response to Treatment",
            ['Excellent', 'Indeterminate', 'Biochemical Incomplete', 'Structural Incomplete']
        )

st.write(" ")  # Spacer
predict_btn = st.button("Analyze Recurrence Risk")
st.write(" ")  # Spacer

# --- Prediction Logic ---
if predict_btn:
    if model is not None:
        # Create input DataFrame matching model features
        # Note: Ensure the column names exactly match what the model was trained on
        input_data = pd.DataFrame([{
            'Hx Radiothreapy': hx_rad,
            'Thyroid Function': thyroid_func,
            'Physical Examination': phys_exam,
            'Risk': risk,
            'Stage': stage,
            'Response': response
        }])

        # Get Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        # Results Display
        st.subheader("📊 Diagnostic Result")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            # Check for both string "Yes" or integer 1 depending on how your model was saved
            if str(prediction) == "Yes" or prediction == 1:
                st.error("### Result: RECURRED")
                score = probability[1]
            else:
                st.success("### Result: NO RECURRENCE")
                score = probability[0]

            st.metric(label="Model Confidence", value=f"{score * 100:.2f}%")

        with res_col2:
            st.write("**Risk Probability Visualization**")
            # Using a progress bar as a visual scale
            st.progress(float(score))
            st.info(
                f"The model is {score * 100:.1f}% certain of this classification based on historical patient data patterns.")
    else:
        st.error("Model not found. Please check the file path.")

# --- Footer ---
st.divider()
st.caption("© 2026 Thyroid Cancer recurrence AI Systems | For Research Use Only | Not a substitute for professional medical advice. Made by Sumon Ray")