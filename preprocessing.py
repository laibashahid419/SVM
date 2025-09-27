import pandas as pd 
import re 

def clean_data(file_path, output_path):
    df = pd.read_csv(file_path, encoding="latin-1", header=None)
    df.columns = ["label", "id", "date", "query", "user", "text"]

    
    df = df[["label", "text"]]

    
    df["text"] = df["text"].apply(lambda x: re.sub(r"http\S+|www\S+", "", str(x)))
    df["text"] = df["text"].str.replace(r"[^a-zA-Z\s]", "", regex=True)
    df["text"] = df["text"].str.lower().str.strip()

    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved at {output_path}")

if __name__ == "__main__":
    clean_data("data/data.csv", "data/cleaned_data.csv")

