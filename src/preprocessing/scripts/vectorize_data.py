import os

import joblib
import numpy as np
import pandas as pd

from .. import get_tfidf_vectorizer
from ..utils import get_csv_files

if __name__ == "__main__":
    preprocessed_data_paths = get_csv_files("data/preprocessed/")
    corpus = []
    
    for path in preprocessed_data_paths:
        assert os.path.exists(path), f"{path!r} doesn't exist!"
        corpus.extend(
            pd.read_csv(path)["text"].to_list()
        )
    
    corpus = np.array(corpus)
    vectorizer = get_tfidf_vectorizer(corpus, save=True)
    vectors = vectorizer.transform(corpus)
    joblib.dump(vectors, "data/vectorized/vectors.pkl")
