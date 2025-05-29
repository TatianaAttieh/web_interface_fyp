import streamlit as st
import joblib
import numpy as np

# === LOAD MODELS AND ENCODERS ===
tfidf = joblib.load("tfidf_vectorizer.pkl")
lr_bin = joblib.load("lr_tfidf_binary.pkl")
lr_multi = joblib.load("lr_tfidf_multiclass.pkl")
le_bin = joblib.load("label_encoder_binary.pkl")
le_multi = joblib.load("label_encoder_multiclass.pkl")

# === SEVERITY SCORE FUNCTION ===
def get_anxiety_severity(proba, classes):
    try:
        class_list = list(classes)
        idx = list(map(str.lower, class_list)).index('anxiety')  # Handle case-insensitive match
        return round(proba[0][idx] * 100, 2)
    except ValueError:
        return None

# === STREAMLIT INTERFACE ===
st.set_page_config(page_title="Mental Health Classifier", layout="centered")

st.title("🧠 Mental Health Text Classifier")
st.write("This app classifies your text into one of the following categories:")
st.markdown("- Normal\n- Anxiety\n- Depression\n- Suicidal")
st.write("It also gives a severity score if the detected status is Anxiety.")

user_input = st.text_area("Enter your text here:", height=150)

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter some text before clicking classify.")
    else:
        # Vectorize input
        X_input = tfidf.transform([user_input])

        # Stage 1: Binary classification
        y_bin_pred = lr_bin.predict(X_input)
        label_bin = le_bin.inverse_transform(y_bin_pred)[0]

        if label_bin.lower() == "normal":
            st.success("✅ The input is classified as NORMAL.")
        else:
            # Stage 2: Multiclass classification
            y_multi_pred = lr_multi.predict(X_input)
            y_multi_proba = lr_multi.predict_proba(X_input)
            label_multi = le_multi.inverse_transform(y_multi_pred)[0]

            st.error(f"⚠️ The input is classified as {label_multi.upper()}.")

            if label_multi.lower() == "anxiety":
                severity = get_anxiety_severity(y_multi_proba, le_multi.classes_)
                if severity is not None:
                    st.info(f" Anxiety Severity Score: {severity}%")