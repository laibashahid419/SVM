import pandas as pd
from src.utils import load_model

def predict_text(text, model_path="models/sentiment_model.pkl", vectorizer_path="models/vectorizer.pkl"):

    model, vectorizer = load_model(model_path, vectorizer_path)
    X_vec = vectorizer.transform([text]) 
    
    prediction = model.predict(X_vec)[0]
    return prediction


