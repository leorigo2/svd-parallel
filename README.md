# svd-parallel

Official repository of the project: **"Parallelization of the Single Value Decomposition algorithm on MPI cluster"**.

This project compares a standard serial implementation of the SVD algorithm against a parallelized version designed for an MPI cluster.

## Repository Structure

### Algorithms
Each algorithm has its own dedicated folder containing:
* **Serial Algorithm (C):** The starting point implementation.
* **Parallel Algorithm (MPI Library):** The optimized version used for evaluation.

### Results
The results subfolders contain:
* Execution times for each matrix and number of processes.
* Graphs comparing performance.

### Key Files
* `dataset.txt`: Contains the dataset of 30 matrices used for testing.
* `generator.py`: A Python script to generate random matrices with different values and dimensions.
* `final_report.pdf`: The complete project report detailing methods and outcomes.

## Overview
We used the serial algorithm in C as a baseline to develop the parallel version. The performance analysis focuses on the execution time differences between the two approaches across the dataset.

## Instructions

To run a test on the UniTN cluster

1. Move to the desired folder
2. Compile the source code with 
    ```bash
    mpicc -g -Wall -std=c99 -o svd_qr_parallel svd_qr_parallel.c -lm
3. Customize the PBS parameters in run.sh based on your test requirements
4. Submit the work by running
    ```bash
    qsub run.sh
