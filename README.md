# CSCI 5451: Introduction to Parallel Computing

My coursework for CSCI 5451 in Spring 2023 at the University of Minnesota. The course covers parallel computer architectures, parallel algorithm analysis, and parallel programming for distributed memory with MPI, shared memory with OpenMP, and GPUs with CUDA. Everything here is written in C or CUDA and is intended to be run on Linux.

Course website: <https://kauffman77.github.io/5451/>


## Repository layout

| Path | Contents |
|---|---|
| [`a2-code/`](a2-code/) | A2: MPI programming. A heat transfer simulation and k-means clustering on a compute cluster |
| [`a3-code/`](a3-code/) | A3: shared memory and CUDA. K-means clustering with OpenMP and on a GPU |
| [`a2-code.zip`](a2-code.zip), [`a3-code.zip`](a3-code.zip) | The original submissions, including the MNIST data |
| [`kmeans-code/sample-mnist-data/`](kmeans-code/sample-mnist-data/) | Two of the MNIST digit files used as k-means input (copies of files in the zips) |


## Assignments

### A2: MPI programming

[`a2-code/`](a2-code/) contains three programs. I ran the timing tests on the Mesabi cluster at the Minnesota Supercomputing Institute (MSI).

**Heat transfer (`heat_mpi`).** A finite difference simulation of the heat equation on a 1D rod with Dirichlet boundary conditions. Each MPI rank is given an equal slice of the rod and neighboring processes exchange their edge cells at each time step. Neighboring ranks communicate via `MPI_Sendrecv` and the root process collects the rows with `MPI_Gather` to output progress.

**Serial k-means (`kmeans_serial`).** A C port of the provided Python k-means program. It clusters MNIST handwritten digits (28x28 images, one per line), prints a confusion matrix of true labels against clusters, and saves each cluster center as a PGM image.

**MPI k-means (`kmeans_mpi`).** A distributed-memory version of the serial program. The root process reads the data and divides the rows among the processes with `MPI_Scatterv`. In each iteration of the k-means algorithm, every process sums the points in its set and `MPI_Allreduce` combines those sums into the new cluster centers. Each process then reassigns its own points, and another `MPI_Allreduce` totals the changed assignments so every process agrees on when to terminate. `MPI_Gatherv` collects the assignments on the root to handle output at termination.

- **Files I wrote:** `heat_mpi.c`, `kmeans_serial.c`, `kmeans_mpi.c`, and `A2-WRITEUP.txt` (timing tables and analysis). `kmeans_mpi2.c` is an earlier draft.
- **Topics:** domain decomposition, boundary exchange between neighbors, collective communication, uneven data partitioning, batch jobs with SLURM
- **Run:** `mpirun -np 4 ./heat_mpi 10 20 1` (time steps, rod width, print flag) and `mpirun -np 4 ./kmeans_mpi mnist-data/digits_all_5e3.txt 20 outdir 500` (data file, clusters, output folder, maximum iterations)

### A3: Shared memory and CUDA

[`a3-code/`](a3-code/) parallelizes the same k-means algorithm in two more ways. I timed both on the CSE Labs CUDA machines.

**OpenMP (`kmeans_omp`).** Threads split up the data points. When computing cluster centers, each cluster's running sum is protected by its own mutex lock. During each assignment step, an OpenMP reduction counts changes and each thread tallies cluster sizes in its own section of an array that is added up afterward. This avoids atomic updates inside the loop.

**CUDA (`kmeans_cuda`).** The data is copied to the GPU once and stays there for the whole run. Center calculation use one block per cluster and one thread per pixel so that every thread writes to a different memory address, removing the need for synchronization. The assignment steps one thread per data point using `atomicAdd` to update cluster and change counts.

Results from [`A3-WRITEUP.txt`](a3-code/A3-WRITEUP.txt) (20 clusters, times in seconds):

| Data file | Images | OpenMP, 1 thread | OpenMP, 32 threads | Speedup | Serial CPU | GPU | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `digits_all_5e3.txt` | 5,000 | 4.43 | 1.15 | 3.9x | 4.21 | 1.84 | 2.3x |
| `digits_all_1e4.txt` | 10,000 | 25.31 | 3.07 | 8.2x | 23.99 | 3.97 | 6.0x |
| `digits_all_3e4.txt` | 30,000 | 69.13 | 8.50 | 8.1x | 65.90 | 9.33 | 7.1x |

- **Files I wrote:** `kmeans_omp.c`, `kmeans_cuda.cu`, `kmeans_serial.c` (my A2 port), and `A3-WRITEUP.txt`
- **Topics:** OpenMP parallel loops, locks, reductions, per-thread buffers to avoid contention, CUDA kernels, grids and blocks, host-to-device memory transfers, atomic operations
- **Run:** `OMP_NUM_THREADS=8 ./kmeans_omp mnist-data/digits_all_5e3.txt 20 outdir 500` and `./kmeans_cuda mnist-data/digits_all_5e3.txt 20 outdir 500`


## Building and running

The programs require Linux and `gcc`. A2 also needs an MPI implementation such as Open MPI (`mpicc` and `mpirun`). `kmeans_cuda` needs an NVIDIA GPU and the CUDA toolkit (`nvcc`). Many tests use `valgrind`. The programs read MNIST data that the folders here leave out, so unzip the original first to add it:

```bash
unzip -n a3-code.zip        # adds mnist-data/ to a3-code/
cd a3-code
make                        # build
make test                   # run all tests
make test-prob1 testnum=3   # run one test
```

Without `nvcc`, build A3's CPU programs with `make kmeans_serial kmeans_omp`. The SLURM scripts in A2 (`heat-slurm.sh`, `kmeans-slurm.sh`) were written for MSI's Mesabi cluster.


## Acknowledgments

- Chris Kauffman wrote the assignments, the starter code (including the serial heat simulation and the Python k-means program), and the GPLv3 `testy` runner.
- The digit images come from the MNIST database by Yann LeCun, Corinna Cortes, and Christopher Burges.
