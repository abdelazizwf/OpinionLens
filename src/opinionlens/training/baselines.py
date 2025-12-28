import random
from collections import Counter

import mlflow
import pandas as pd

from opinionlens.training.utils import calculate_metrics


def zero_rule_baseline(train_data: pd.DataFrame, test_data: pd.DataFrame):
    scores = train_data["score"]
    most_common_score = Counter(scores).most_common(1)[0][0]
    predictions = [most_common_score] * len(test_data)
    return predictions

def random_baseline(train_data: pd.DataFrame, test_data: pd.DataFrame):
    return [random.randint(0, 1) for _ in range(len(test_data))]


def heuristic_baseline(train_data: pd.DataFrame, test_data: pd.DataFrame):
    good_words = "good great best enjoyed loved :) masterpiece".split()
    bad_words = "bad worst shit crap hated :( repugnant boring shallow".split()

    predictions = []
    for _, text in test_data["text"].items():
        word_counter = Counter(text.lower().split())
        good_word_count = sum(word_counter.get(word, 0) for word in good_words)
        bad_word_count = sum(word_counter.get(word, 0) for word in bad_words)
        predictions.append(1 if good_word_count >= bad_word_count else 0)

    return predictions


def main():
    train_data = pd.read_csv("./data/preprocessed/imdb_dataset/train.csv")
    test_data = pd.read_csv("./data/preprocessed/imdb_dataset/test.csv")

    with mlflow.start_run(run_name="baselines"):
        for func in [
            zero_rule_baseline, random_baseline, heuristic_baseline,
        ]:
            with mlflow.start_run(run_name=func.__name__, nested=True):
                predictions = func(train_data, test_data)
                truths = test_data["score"].to_list()
                assert len(truths) == len(predictions), f"{len(truths)} != {len(predictions)}"

                metrics, con_matrix_fig, roc_fig = calculate_metrics(
                    truths, predictions, prefix="test_", figures=True
                )

                mlflow.log_metrics(metrics)
                mlflow.log_figure(con_matrix_fig, "figures/confusion_matrix.png")
                mlflow.log_figure(roc_fig, "figures/roc_curve.png")


if __name__ == "__main__":
    main()
