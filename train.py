import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def clean_text(text):
    import re, string
    text = text.lower()
    text = re.sub('\[.*?\]', "", text)
    text = re.sub("\\W", " ", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text

def retrain_models(csv_path="cleaned_news_labels.csv"):
    df = pd.read_csv(csv_path)

    if 'text' not in df.columns or 'true_label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'true_label' columns")

    df['text'] = df['text'].apply(clean_text)
    x = df['text']
    y = df['true_label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    xv_train = vectorizer.fit_transform(x_train)
    xv_test = vectorizer.transform(x_test)

    # Define models
    models = {
        "logistic_regression": LogisticRegression(),
        "naive_bayes": MultinomialNB(),
        "random_forest": RandomForestClassifier(),
        "gradient_boosting": GradientBoostingClassifier(),
        "svm": SVC(probability=True)
    }

    accuracies = {}

    for name, model in models.items():
        model.fit(xv_train, y_train)
        acc = model.score(xv_test, y_test)
        joblib.dump(model, f"{name}.jb")
        accuracies[name.replace("_", " ").title()] = round(acc * 100, 2)

    joblib.dump(vectorizer, "vectorizer.jb")
    joblib.dump(accuracies, "model_accuracies.jb")

    return accuracies
