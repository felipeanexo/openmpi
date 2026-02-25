import numpy as np

__all__ = [
    "config_error",
    "header",
    "startup",
    "parallel_results",
    "verification",
    "stats",
]

OUTPUT_WIDTH = 80
MAX_DIVISORS_SHOWN = 20


def config_error(n: int, size: int) -> None:
    valid_divisors = [i for i in range(1, n + 1) if n % i == 0]
    print(f"ERROR: Matrix size ({n}) is not divisible by number of processes ({size}).")
    print(f"Use a process count that divides {n}.")
    print(f"Valid divisors: {valid_divisors[:MAX_DIVISORS_SHOWN]}")


def header(n: int, size: int) -> None:
    print("=" * OUTPUT_WIDTH)
    print("PARALLEL MATRIX MULTIPLICATION WITH MPI")
    print("=" * OUTPUT_WIDTH)
    print(f"Matrix size: {n} x {n}")
    print(f"MPI processes: {size}")
    print("=" * OUTPUT_WIDTH)


def startup(rank: int, n: int) -> None:
    print(f"\n[Rank {rank}] Initializing matrices A and B...")
    print(f"[Rank {rank}] Matrices created successfully.")
    print(f"[Rank {rank}] Matrix A: ({n}, {n}), Matrix B: ({n}, {n})")
    print(f"\n{'-' * OUTPUT_WIDTH}")
    print("STARTING PARALLEL COMPUTATION...")
    print(f"{'-' * OUTPUT_WIDTH}\n")


def parallel_results(C_parallel: np.ndarray, parallel_time: float) -> None:
    print("=" * OUTPUT_WIDTH)
    print("PARALLEL EXECUTION RESULTS")
    print("=" * OUTPUT_WIDTH)
    print(f"Parallel execution time: {parallel_time:.6f} seconds")
    print(f"Result matrix C shape: {C_parallel.shape}")


def verification(
    C_parallel: np.ndarray,
    C_sequential: np.ndarray,
    sequential_time: float,
    tolerance: float,
) -> None:
    print(f"\n{'-' * OUTPUT_WIDTH}")
    print("VERIFICATION: Parallel vs sequential comparison")
    print(f"{'-' * OUTPUT_WIDTH}\n")
    print(f"Sequential execution time: {sequential_time:.6f} seconds")

    are_equal = bool(np.allclose(C_parallel, C_sequential, atol=tolerance, rtol=tolerance))
    print("\n" + "=" * OUTPUT_WIDTH)
    if are_equal:
        print("[OK] SUCCESS: Parallel result is mathematically correct.")
        print("     (Parallel result matches sequential reference.)")
    else:
        max_diff = float(np.max(np.abs(C_parallel - C_sequential)))
        print("[ERROR] Results do not match.")
        print(f"  Maximum difference: {max_diff}")
    print("=" * OUTPUT_WIDTH)


def stats(
    n: int,
    size: int,
    rows_per_process: int,
    parallel_time: float,
    sequential_time: float,
) -> None:
    print("\n" + "=" * OUTPUT_WIDTH)
    print("FINAL STATISTICS")
    print("=" * OUTPUT_WIDTH)
    print(f"Matrix size (N x N):        {n} x {n}")
    print(f"MPI processes:              {size}")
    print(f"Rows per process:           {rows_per_process}")
    print(f"Parallel time:              {parallel_time:.6f} s")
    print(f"Sequential time:            {sequential_time:.6f} s")
    if parallel_time > 0:
        speedup = sequential_time / parallel_time
        efficiency_pct = (speedup / size) * 100
        print(f"Speedup:                    {speedup:.2f}x")
        print(f"Efficiency:                 {efficiency_pct:.2f}%")
    print("=" * OUTPUT_WIDTH)
    print("\nRun completed successfully.")
    print("=" * OUTPUT_WIDTH)
