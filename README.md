# Parallel matrix multiplication with OpenMPI

Parallel matrix multiplication using **MPI** (OpenMPI) and **mpi4py**. The workload is distributed across processes with collective operations: **Bcast**, **Scatter**, and **Gather**.

## Requirements

- **Python** ≥ 3.12  
- **OpenMPI** (system): `mpirun` and MPI libraries  
- **uv** (recommended) or pip + venv  

### Install OpenMPI (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y openmpi-bin libopenmpi-dev
```

## Setup

From the project root:

```bash
uv sync
```

## Run

Use a number of processes that **divides** the matrix size (see `src/config.py`). Example with 4 processes:

```bash
mpirun -n 4 uv run python main.py
```

Or using the venv Python directly:

```bash
mpirun -n 4 .venv/bin/python main.py
```

## Configuration

Edit `src/config.py`:

| Variable         | Default  | Description                    |
|------------------|----------|--------------------------------|
| `MATRIX_SIZE`    | `10_000` | Matrix dimension (N×N)         |
| `SEED`           | `42`     | Random seed for reproducibility |
| `FLOAT_TOLERANCE`| `1e-10`  | Tolerance for result comparison |

**Note:** Very large sizes (e.g. 1M×1M) require ~7.3 TB per matrix and are not feasible on typical machines. Use 10_000–20_000 for large runs on 8–32 GB RAM.

## Project layout

```
openmpi/
├── main.py              # Entrypoint: MPI init, validation, orchestration
├── pyproject.toml
└── src/
    ├── config.py        # MATRIX_SIZE, SEED, FLOAT_TOLERANCE
    ├── data.py          # create_matrices(), sequential_multiply(), results_match()
    ├── parallel.py      # Bcast / Scatter / Gather, run_parallel_multiplication()
    └── report.py        # Console output and statistics
```

## MPI concepts used

- **Rank 0** builds matrices A and B and coordinates the run.  
- **Bcast:** matrix B is sent from rank 0 to all processes (everyone needs full B).  
- **Scatter:** matrix A is split into row chunks; each process gets its chunk.  
- **Gather:** each process sends its partial result to rank 0, which assembles the full C.  
- Result is checked against a sequential NumPy multiplication on rank 0.

## License

Academic use.
