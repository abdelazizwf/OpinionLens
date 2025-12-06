import os

import pandas as pd
from omegaconf import OmegaConf

from .. import clean_text, tokenizer

conf = OmegaConf.load("./params.yaml")


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
    imdb_data = imdb_data.sample(
        frac=1, random_state=conf.base.random_seed
    ).reset_index(drop=True)
    
    train_frac, val_frac, test_frac = conf.preprocessing.data_splits
    preprocessed_data_path = "data/preprocessed/imdb_dataset/"
    if not os.path.exists(preprocessed_data_path):
        os.makedirs(preprocessed_data_path)
    
    train_index = int((train_frac * len(imdb_data)))
    val_index = int((val_frac * len(imdb_data)) + train_index)
    test_index = int((test_frac * len(imdb_data)) + val_index)
    
    imdb_data.iloc[:train_index - 1].to_csv(os.path.join(preprocessed_data_path, "train.csv"), index=False)
    imdb_data.iloc[train_index:val_index - 1].to_csv(os.path.join(preprocessed_data_path, "val.csv"), index=False)
    imdb_data.iloc[val_index:test_index - 1].to_csv(os.path.join(preprocessed_data_path, "test.csv"), index=False)
    

if __name__ == "__main__":
    preprocess_imdb_dataset()
