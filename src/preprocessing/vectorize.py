import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

SAVED_VECTORIZER_PATH = "./objects/vectorizer.pkl"


def get_tfidf_vectorizer(training_corpus, save=False):
    vectorizer = TfidfVectorizer(
        strip_accents=None, lowercase=False, preprocessor=None, tokenizer=None
    )
    vectorizer.fit(training_corpus)
    
    if save:
        joblib.dump(vectorizer, SAVED_VECTORIZER_PATH)
    
    return vectorizer


def get_saved_tfidf_vectorizer():
    assert os.path.exists(SAVED_VECTORIZER_PATH), f"{SAVED_VECTORIZER_PATH!r} doesn't exist!"
    vectorizer = joblib.load(SAVED_VECTORIZER_PATH)
    return vectorizer


if __name__ == "__main__":
    import numpy as np
    
    corpus = np.array(["Hello from here it is", "Hello here I am"])
    vectorizer = get_tfidf_vectorizer(corpus)
    print(vectorizer.transform(corpus).toarray())
