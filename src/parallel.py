import time
import numpy as np
from mpi4py import MPI

__all__ = ["run_parallel_multiplication"]


def _broadcast_B(comm: MPI.Intracomm, B: np.ndarray | None, n: int, rank: int) -> np.ndarray:
    if rank != 0:
        B = np.empty((n, n), dtype=np.float64)
    comm.Bcast(B, root=0)
    return B


def _scatter_A(
    comm: MPI.Intracomm,
    A: np.ndarray | None,
    n: int,
    size: int,
    rank: int,
) -> np.ndarray:
    rows_per_process = n // size
    if rank == 0:
        A_to_scatter = A.reshape(size, rows_per_process, n)
    else:
        A_to_scatter = None
    A_local = np.empty((rows_per_process, n), dtype=np.float64)
    comm.Scatter(A_to_scatter, A_local, root=0)
    return A_local


def _local_multiply(A_local: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.dot(A_local, B)


def _gather_C(
    comm: MPI.Intracomm,
    C_local: np.ndarray,
    n: int,
    size: int,
    rank: int,
) -> np.ndarray | None:
    rows_per_process = n // size
    if rank == 0:
        C_gathered = np.empty((size, rows_per_process, n), dtype=np.float64)
    else:
        C_gathered = None
    comm.Gather(C_local, C_gathered, root=0)
    if rank == 0:
        return C_gathered.reshape(n, n)
    return None


def run_parallel_multiplication(
    comm: MPI.Intracomm,
    A: np.ndarray | None,
    B: np.ndarray | None,
    n: int,
) -> tuple[np.ndarray | None, float]:
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        start = time.perf_counter()

    B = _broadcast_B(comm, B, n, rank)
    A_local = _scatter_A(comm, A, n, size, rank)
    C_local = _local_multiply(A_local, B)
    C_parallel = _gather_C(comm, C_local, n, size, rank)

    if rank == 0:
        elapsed = time.perf_counter() - start
        return C_parallel, elapsed
    return None, 0.0
