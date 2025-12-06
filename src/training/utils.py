import joblib


def load_data():
    X_train = joblib.load("data/vectorized/train_vectors.pkl")
    X_val = joblib.load("data/vectorized/val_vectors.pkl")
    X_test = joblib.load("data/vectorized/test_vectors.pkl")
    
    y_train = joblib.load("data/vectorized/train_scores.pkl")
    y_val = joblib.load("data/vectorized/val_scores.pkl")
    y_test = joblib.load("data/vectorized/test_scores.pkl")
    
    return X_train, X_val, X_test, y_train, y_val, y_test