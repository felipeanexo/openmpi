# Parallel matrix multiplication with OpenMPI

Parallel matrix multiplication using **MPI** (OpenMPI) and **mpi4py**. The workload is distributed across processes with collective operations: **Bcast**, **Scatter**, and **Gather**.

---

## What is MPI?

**MPI** (Message Passing Interface) is a **standard API for parallel programming in distributed-memory systems**. It defines how separate processes can exchange data and synchronize by sending and receiving messages. It is the dominant model for programming clusters, multi-node HPC systems, and any environment where each process has its own memory and must communicate explicitly with others.

### Why “message passing”?

- In **shared memory** (e.g. one machine with many cores), threads or processes can read/write the same RAM; synchronization is done with locks, barriers, etc. (e.g. OpenMP, pthreads).
- In **distributed memory** (e.g. several machines, or processes that do not share address space), each process has its **own memory**. One process cannot directly read another’s variables. To share data, they must **send and receive messages** (e.g. “here is an array”, “give me your result”). MPI is the standard that defines how to do this in a portable way.

So: **MPI = agreed set of functions and rules for communication and synchronization between processes that do not share memory.**

### Main concepts

- **Processes:** The program is launched as **multiple independent processes** (e.g. 4 or 100). Each has its own memory space. They may run on one machine (multi-core) or on many machines (cluster).
- **Communicator:** A group of processes that can talk to each other. The usual one is `MPI_COMM_WORLD`: all processes started together for this program.
- **Rank:** Inside a communicator, each process has a unique **rank**: an integer id (0, 1, 2, …). Rank is the “name” used in send/receive and collective calls (e.g. “send to rank 3”, “root is rank 0”).
- **Size:** Number of processes in the communicator (e.g. 4 if you ran `mpirun -n 4`).

So: **same program runs on every process**; each process uses `rank` and `size` to decide “am I the coordinator?” or “which chunk of work is mine?” and to choose the right communication calls.

### Point-to-point vs collective

- **Point-to-point:** One process sends to one other (e.g. `Send` / `Recv`). Flexible, but you must pair every send with a receive and manage who talks to whom.
- **Collective:** All processes in the communicator participate in one call; the library coordinates who sends what. Examples:
  - **Broadcast (Bcast):** one process (root) sends the same data to everyone.
  - **Scatter:** root splits one buffer and sends a different piece to each process.
  - **Gather:** each process sends a piece to the root; root concatenates into one buffer.
  - **Barrier:** all processes wait until everyone has reached the barrier.

This project uses **only collective** operations (Bcast, Scatter, Gather), which keeps the program simple and avoids explicit pairing of sends/receives.

### SPMD

Typical MPI style is **SPMD** (Single Program, Multiple Data): the **same binary/script** runs on every process. Behavior differs by rank (e.g. `if rank == 0: create data; else: receive data`). So one source code, many processes, each with a different role or slice of data.

### OpenMPI and mpi4py

- **MPI** is a **standard** (specification). Different implementations exist: **OpenMPI**, MPICH, Microsoft MPI, etc. They all offer the same MPI API (with minor extras).
- **OpenMPI** is one such implementation: it provides the MPI library and the **mpirun** command that starts your program with many processes (local or across nodes).
- **mpi4py** is a **Python binding** to MPI: it lets you call MPI from Python (send/receive, collective, communicators, etc.). At runtime it uses whatever MPI implementation is installed (e.g. OpenMPI); that’s why you need both: `pip install mpi4py` (Python side) and system OpenMPI (actual MPI runtime and `mpirun`).

### When to use MPI

- **Multi-node clusters:** several machines; processes on different nodes cannot share memory → message passing (MPI) is natural.
- **Large problems:** when data or work is too big for one node, you split across processes and use MPI to coordinate.
- **Distributed memory by design:** even on one machine, if you want independent processes (e.g. one Python interpreter per process, clear memory boundaries), MPI gives a portable, standard way to communicate.

Compared to **OpenMP** (shared memory, one node, compiler directives): MPI is for **distributed memory** and multi-node; OpenMP is for **shared memory** on a single node. They can be combined (e.g. MPI between nodes, OpenMP inside a node).

---

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
