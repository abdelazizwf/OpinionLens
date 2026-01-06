import sys
from collections import Counter

import mlflow
import pandas as pd
from omegaconf import OmegaConf

from opinionlens.preprocessing.vectorize import get_saved_tfidf_vectorizer
from opinionlens.training.utils import calculate_metrics

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


def main():
    model_id = sys.argv[1]
    model = mlflow.pyfunc.load_model(f"models:/{model_id}")

    data = pd.read_csv("./data/preprocessed/amazon_food_reviews/test.csv")
    balanced_data = get_balanced_data(data)
    short_text, long_text = get_short_and_long_text(data)
    less_common, more_common = get_text_with_common_words(data)

    runs = [
        (balanced_data, "balanced_data"),
        (short_text, "short_text_data"),
        (long_text, "long_text_data"),
        (less_common, "less_common_words_data"),
        (more_common, "more_common_words_data"),
    ]

    vectorizer = get_saved_tfidf_vectorizer()

    with mlflow.start_run(run_name="evals"):
        mlflow.log_param("model_id", model_id)

        for data, run_name in runs:
            vectors = vectorizer.transform(data["text"])

            with mlflow.start_run(nested=True, run_name=run_name):
                predictions = model.predict(vectors)

                metrics, con_matrix_fig, roc_fig = calculate_metrics(
                    data["score"], predictions, prefix="test_", figures=True
                )

                mlflow.log_metrics(metrics)
                mlflow.log_figure(con_matrix_fig, "figures/confusion_matrix.png")
                mlflow.log_figure(roc_fig, "figures/roc_curve.png")


if __name__ == "__main__":
    main()
