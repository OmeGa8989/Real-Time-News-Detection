import streamlit as st
import joblib
from train import retrain_models  # ✅ Import retraining logic

# --- Load Vectorizer & Accuracies ---
vectorizer = joblib.load("vectorizer.jb")
accuracies = joblib.load("model_accuracies.jb")

# --- Define Available Models ---
models = {
    "Logistic Regression": "logistic_regression.jb",
    "SVM": "svm.jb",
    "Naïve Bayes": "naive_bayes.jb",
    "Random Forest": "random_forest.jb",
    "Gradient Boosting": "gradient_boosting.jb"
}

# --- Streamlit UI ---
st.set_page_config(page_title="News Legitimacy Detector", layout="centered")
st.title("📰 News Legitimacy Detector")
st.write("Enter real-time news text to check whether it's real or fake.")

# --- Input Section ---
news_input = st.text_area("✏️ Enter News Text:", "")
model_choice = st.selectbox("🤖 Choose a Model:", list(models.keys()))

# --- Prediction Logic ---
if st.button("Check"):
    if news_input.strip():
        model = joblib.load(models[model_choice])
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)
        accuracy = accuracies.get(model_choice, 0)

        if prediction[0] == 1:
            st.success(f"✅ Verified News (Accuracy: {accuracy:.2f}%)")
        else:
            st.error(f"❌ Fake News Detected (Accuracy: {accuracy:.2f}%)")
    else:
        st.warning("⚠️ Please enter some text to analyze!")

# --- Section: Retrain Models ---
st.markdown("---")
st.header("🔄 Retrain Models with New Data")

if st.button("Retrain Models"):
    try:
        accuracies = retrain_models("cleaned_news_labels.csv")
        st.success("✅ Models retrained successfully!")
        st.write("### 🧠 Updated Model Accuracies")
        for model, acc in accuracies.items():
            st.write(f"- **{model}**: {acc:.2f}%")
    except Exception as e:
        st.error(f"❌ Retraining failed: {str(e)}")
