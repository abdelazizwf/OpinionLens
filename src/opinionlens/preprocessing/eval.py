from collections import Counter

import pandas as pd
from omegaconf import OmegaConf

conf = OmegaConf.load("params.yaml")


def get_balanced_data(df: pd.DataFrame) -> pd.DataFrame:
    score_groups = df.groupby("score")
    min_count = score_groups.count().min().item()
    balanced_data = score_groups.sample(min_count, random_state=conf.base.random_seed)
    return balanced_data


def get_short_and_long_text(df: pd.DataFrame, q: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    df["text_len"] = df["text"].apply(len)
    text_len_threshold = df["text_len"].quantile(q)

    short_text = df.loc[df["text_len"] < text_len_threshold, ["text", "score"]]
    long_text = df.loc[df["text_len"] >= text_len_threshold, ["text", "score"]]

    return short_text, long_text


def get_text_with_common_words(data: pd.DataFrame, q: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    counter = Counter()
    data["text"].apply(lambda x: counter.update(Counter(x.split())))
    occurence_count = sum(counter.values()) # Simplifying constant
    data["commonality_score"] = data["text"].apply(
        lambda x: sum(counter[word] for word in x.split()) / occurence_count
    )

    commonality_threshold = data["commonality_score"].quantile(q)
    less_common = data.loc[data["commonality_score"] < commonality_threshold, ["text", "score"]]
    more_common = data.loc[data["commonality_score"] >= commonality_threshold, ["text", "score"]]

    return less_common, more_common
