import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def train_model(data_path, model_path, vectorizer_path):
    df = pd.read_csv(data_path, encoding="latin-1", header=None)
    df.columns = ["label", "id", "date", "query", "user", "text"]

    X = df["text"]
    y = df["label"]


    vectorizer = CountVectorizer(stop_words="english", max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

   
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

   
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f" Model saved at {model_path}")
    print(f" Vectorizer saved at {vectorizer_path}")


if __name__ == "__main__":
    train_model("data/data.csv", "models/sentiment_model.pkl", "models/vectorizer.pkl")

