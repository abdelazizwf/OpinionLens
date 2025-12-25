from prometheus_client import CollectorRegistry, Counter, Histogram

inference_registry = CollectorRegistry()

MODEL_INFERENCE_TIME_SECONDS = Histogram(
    "model_inference_time_seconds",
    "Model inference duration",
    ["endpoint", "model_class"],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0),
    registry=inference_registry,
)

INPUT_TEXT_LENGTH_CHARS = Histogram(
    "input_text_length_chars",
    "Input text length in characters",
    ["endpoint"],
    buckets=(10, 50, 100, 200, 500, 1000, 2000),
    registry=inference_registry,
)

BATCH_SIZE_TEXT = Histogram(
    "batch_size_text",
    "Number of texts in batch",
    ["endpoint"],
    buckets=(2, 5, 10, 20, 50, 100, 200, 500, 1000),
    registry=inference_registry,
)

BATCH_INFERENCE_TIME_PER_ITEM_SECONDS = Histogram(
    "batch_inference_time_per_item_seconds",
    "Inference time per item in batch",
    ["endpoint", "model_class"],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0),
    registry=inference_registry,
)

PREDICTED_SENTIMENT_TOTAL = Counter(
    "predicted_sentiment_total",
    "Predicted sentiment classes",
    ["label"],
    registry=inference_registry,
)
