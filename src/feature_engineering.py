from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

def get_vectorizer():
    return TfidfVectorizer(max_features=5000)

def fit_vectorizer(texts, save_path="models/vectorizer.pkl"):
    vectorizer = get_vectorizer()
    X = vectorizer.fit_transform(texts)
    
    os.makedirs("models", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(vectorizer, f)
    
    return vectorizer, X

def load_vectorizer(path="models/vectorizer.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def transform_text(vectorizer, texts):
    return vectorizer.transform(texts)