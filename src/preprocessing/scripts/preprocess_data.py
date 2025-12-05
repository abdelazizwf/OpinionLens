import os

import pandas as pd

from .. import clean_text, tokenizer


def tokenize_text(text):
    return " ".join(tokenizer(clean_text(text)))


def preprocess_imdb_dataset():
    raw_data_path = "data/raw/IMDB Dataset/IMDB Dataset.csv"
    assert os.path.exists(raw_data_path), f"{raw_data_path!r} doesn't exist!"
    
    imdb_data = pd.read_csv(raw_data_path)
    
    imdb_data["score"] = imdb_data["sentiment"].map(
        {"positive": 1, "negative": 0}
    )
    imdb_data["text"] = imdb_data["review"].apply(tokenize_text)
    imdb_data.drop(columns=["review", "sentiment"], inplace=True)
    
    imdb_data.to_csv("data/preprocessed/imdb_dataset.csv", index=False)


if __name__ == "__main__":
    preprocess_imdb_dataset()
