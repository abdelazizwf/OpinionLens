import os

import pandas as pd
from omegaconf import OmegaConf

from opinionlens.preprocessing import clean_text, tokenizer
from opinionlens.preprocessing.utils import save_preprocessed_data

conf = OmegaConf.load("./params.yaml")


def tokenize_text(text: str) -> str:
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

    preprocessed_data_path = "data/preprocessed/imdb_dataset/"
    save_preprocessed_data(imdb_data, preprocessed_data_path)


def preprocess_amazon_food_dataset():
    raw_data_path = "data/raw/Amazon Food Reviews/Reviews.csv"
    assert os.path.exists(raw_data_path), f"{raw_data_path!r} doesn't exist!"

    data = pd.read_csv(raw_data_path)

    data["score"] = data["Score"].map(
        {1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    )
    data["text"] = data["Text"].apply(tokenize_text)

    data = data[["text", "score"]].sample(
        frac=1, random_state=conf.base.random_seed
    ).reset_index(drop=True)

    preprocessed_data_path = "data/preprocessed/amazon_food_reviews/"
    save_preprocessed_data(data, preprocessed_data_path)


def main():
    preprocess_imdb_dataset()
    preprocess_amazon_food_dataset()


if __name__ == "__main__":
    main()
