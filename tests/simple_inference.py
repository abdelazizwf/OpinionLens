import mlflow
import numpy as np

from src.preprocessing.clean import clean_text
from src.preprocessing.tokenize import tokenizer
from src.preprocessing.vectorize import get_saved_tfidf_vectorizer

if __name__ == "__main__":
    model = mlflow.sklearn.load_model(
        "models:/m-2d338c3efafc472bb2910b01f2947e82"
    )
    
    texts = [
        "This is so boring I'd rather eat a cockroach!", "My mom threw away the remote in excitement.",
        "I liked this product very much!", "I am so in love with this song my ears are bleeding from non-stop listening.",
    ]
    
    scores = [0, 1, 1, 1]
    
    for i, (text, score) in enumerate(zip(texts, scores)):
        tokenized_text = " ".join(tokenizer(clean_text(text)))

        vectorizer = get_saved_tfidf_vectorizer()
        vectors = vectorizer.transform(np.array([tokenized_text]))

        prediction = int(model.predict(vectors)[0])

        assert prediction == score, f"Wrong prediction, got {prediction!r} instead of {score!r} at sample {i!r}."
