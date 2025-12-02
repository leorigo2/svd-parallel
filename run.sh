#/bin/bash

#PBS -l select=4:ncpus=4:mem=2gb
# set max execution time
#PBS -l walltime=0:05:00
# set the queue
#PBS -q short_cpuQ
module load mpich-3.2
mpiexec -n 4 ./svd-parallel/svd_qr.exe 4
