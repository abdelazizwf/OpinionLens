import os

import joblib
import numpy as np
import pandas as pd

from opinionlens.preprocessing import get_tfidf_vectorizer
from opinionlens.preprocessing.utils import get_csv_files


def main():
    preprocessed_data_paths = get_csv_files("data/preprocessed/")
    assert preprocessed_data_paths, "No preprocessed data found!"

    train_corpus = []
    val_corpus = []
    test_corpus = []

    train_scores = []
    val_scores = []
    test_scores = []

    for path in preprocessed_data_paths:
        csv = pd.read_csv(path)
        text = csv["text"].to_list()
        scores = csv["score"].to_list()
        if path.endswith("train.csv"):
            train_corpus.extend(text)
            train_scores.extend(scores)
        elif path.endswith("val.csv"):
            val_corpus.extend(text)
            val_scores.extend(scores)
        elif path.endswith("test.csv"):
            test_corpus.extend(text)
            test_scores.extend(scores)

    vectorizer = get_tfidf_vectorizer(train_corpus, save=True)

    train_vectors = vectorizer.transform(train_corpus)
    val_vectors = vectorizer.transform(val_corpus)
    test_vectors = vectorizer.transform(test_corpus)

    vectors_path = "data/vectorized/"
    os.makedirs(vectors_path, exist_ok=True)

    joblib.dump(train_vectors, os.path.join(vectors_path, "train_vectors.pkl"))
    joblib.dump(val_vectors, os.path.join(vectors_path, "val_vectors.pkl"))
    joblib.dump(test_vectors, os.path.join(vectors_path, "test_vectors.pkl"))

    joblib.dump(np.array(train_scores), os.path.join(vectors_path, "train_scores.pkl"))
    joblib.dump(np.array(val_scores), os.path.join(vectors_path, "val_scores.pkl"))
    joblib.dump(np.array(test_scores), os.path.join(vectors_path, "test_scores.pkl"))


if __name__ == "__main__":
    main()
