import os

import pandas as pd
from omegaconf import OmegaConf

from opinionlens.common.utils import get_csv_files
from opinionlens.preprocessing import clean_text, eval, tokenizer
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


def preprocess_airline_tweets():
    raw_data_path = "data/raw/Airline Tweets/Tweets.csv"
    assert os.path.exists(raw_data_path), f"{raw_data_path!r} doesn't exist!"

    data = pd.read_csv(raw_data_path)

    # Drop neutral tweets
    data = data.loc[data["airline_sentiment"] != "neutral"]

    data["score"] = data["airline_sentiment"].map(
        {"positive": 1, "negative": 0}
    )

    data["text"] = data["text"].apply(tokenize_text)
    data = data[["text", "score"]].sample(
        frac=1, random_state=conf.base.random_seed
    ).reset_index(drop=True)

    preprocessed_data_path= "data/preprocessed/airline_tweets/"
    save_preprocessed_data(data, preprocessed_data_path)


def preprocess_eval_data():
    files = get_csv_files("data/preprocessed/", prefix="test")
    df = pd.DataFrame()
    for file in files:
        df = pd.concat(
            [df, pd.read_csv(file)], axis=0,
        )

    eval_data_path = "data/eval_data/"
    os.makedirs(eval_data_path, exist_ok=True)

    balanced_data = eval.get_balanced_data(df)
    balanced_data.to_csv(os.path.join(eval_data_path, "balanced_data.csv"))

    short_text, long_text = eval.get_short_and_long_text(df)
    short_text.to_csv(os.path.join(eval_data_path, "short_text.csv"))
    long_text.to_csv(os.path.join(eval_data_path, "long_text.csv"))

    less_common, more_common = eval.get_text_with_common_words(df)
    less_common.to_csv(os.path.join(eval_data_path, "less_common_words.csv"))
    more_common.to_csv(os.path.join(eval_data_path, "more_common_words.csv"))


def main():
    preprocess_imdb_dataset()
    preprocess_amazon_food_dataset()
    preprocess_airline_tweets()
    preprocess_eval_data()


if __name__ == "__main__":
    main()
