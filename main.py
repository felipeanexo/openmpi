from mpi4py import MPI

from src.config import MATRIX_SIZE, SEED, FLOAT_TOLERANCE
from src.data import create_matrices, sequential_multiply
from src.parallel import run_parallel_multiplication
from src import report


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        report.header(MATRIX_SIZE, size)
        if MATRIX_SIZE % size != 0:
            report.config_error(MATRIX_SIZE, size)
            comm.Abort()

    A, B = create_matrices(MATRIX_SIZE, SEED) if rank == 0 else (None, None)

    if rank == 0:
        report.startup(rank, MATRIX_SIZE)

    C_parallel, parallel_time = run_parallel_multiplication(comm, A, B, MATRIX_SIZE)

    if rank != 0:
        return

    report.parallel_results(C_parallel, parallel_time)
    C_sequential, sequential_time = sequential_multiply(A, B)
    report.verification(C_parallel, C_sequential, sequential_time, FLOAT_TOLERANCE)
    report.stats(MATRIX_SIZE, size, MATRIX_SIZE // size, parallel_time, sequential_time)


if __name__ == "__main__":
    main()
