__all__ = ["MATRIX_SIZE", "SEED", "FLOAT_TOLERANCE"]

# 1M x 1M would need ~7.3 TB per matrix → OOM on normal machines. Use 10_000–20_000 for "large" runs.
MATRIX_SIZE = 10_000
SEED = 42
FLOAT_TOLERANCE = 1e-10
