import time
import numpy as np

__all__ = ["create_matrices", "sequential_multiply", "results_match"]


def create_matrices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    A = np.random.rand(n, n).astype(np.float64)
    B = np.random.rand(n, n).astype(np.float64)
    return A, B


def sequential_multiply(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    C = np.dot(A, B)
    elapsed = time.perf_counter() - start
    return C, elapsed


def results_match(C_parallel: np.ndarray, C_sequential: np.ndarray, tolerance: float) -> bool:
    return bool(np.allclose(C_parallel, C_sequential, atol=tolerance, rtol=tolerance))
