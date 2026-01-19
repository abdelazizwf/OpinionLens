import os
from typing import Collection

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

SAVED_VECTORIZER_PATH = "./objects/vectorizer.pkl"


def get_tfidf_vectorizer(training_corpus: Collection, save=False) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        strip_accents=None, lowercase=False, preprocessor=None, tokenizer=None
    )
    vectorizer.fit(training_corpus)

    if save:
        os.makedirs(os.path.dirname(SAVED_VECTORIZER_PATH), exist_ok=True)
        joblib.dump(vectorizer, SAVED_VECTORIZER_PATH)

    return vectorizer


def get_saved_tfidf_vectorizer() -> TfidfVectorizer:
    assert os.path.exists(SAVED_VECTORIZER_PATH), f"{SAVED_VECTORIZER_PATH!r} doesn't exist!"
    vectorizer = joblib.load(SAVED_VECTORIZER_PATH)
    return vectorizer
