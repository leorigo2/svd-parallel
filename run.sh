#!/bin/bash

#PBS -l select=4:ncpus=24:mem=2gb 
# set max execution time
#PBS -l walltime=5:00:00
# set the queue
#PBS -q short_HPC4DS
module load mpich-3.2
mpiexec -n 96 ./svd-parallel/svd_qr_parallel 
