import os

import joblib
import numpy as np
import pandas as pd

from .. import get_tfidf_vectorizer
from ..utils import get_csv_files

if __name__ == "__main__":
    preprocessed_data_paths = get_csv_files("data/preprocessed/")
    assert preprocessed_data_paths, "No preprocessed data found!"
    
    train_corpus = []
    val_corpus = []
    test_corpus = []
    
    for path in preprocessed_data_paths:
        text = pd.read_csv(path)["text"].to_list()
        if path.endswith("train.csv"):
            train_corpus.extend(text)
        elif path.endswith("val.csv"):
            val_corpus.extend(text)
        elif path.endswith("test.csv"):
            test_corpus.extend(text)
    
    train_corpus = np.array(train_corpus)
    val_corpus = np.array(val_corpus)
    test_corpus = np.array(test_corpus)
    
    vectorizer = get_tfidf_vectorizer(train_corpus, save=True)
    
    train_vectors = vectorizer.transform(train_corpus)
    val_vectors = vectorizer.transform(val_corpus)
    test_vectors = vectorizer.transform(test_corpus)
    
    joblib.dump(train_vectors, "data/vectorized/train_vectors.pkl")
    joblib.dump(val_vectors, "data/vectorized/val_vectors.pkl")
    joblib.dump(test_vectors, "data/vectorized/test_vectors.pkl")
