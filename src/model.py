from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

def train_classifier(X, y, save_path="models/classifier.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    os.makedirs("models", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    return model, acc


def load_model(path="models/classifier.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_role(model, vectorizer, resume_text):
    vec = vectorizer.transform([resume_text])
    return model.predict(vec)[0]