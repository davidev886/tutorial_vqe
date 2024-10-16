#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpu-bind=none
#SBATCH -t 00:20:00
#SBATCH -q debug
#SBATCH -A m4642
#SBATCH -C gpu
#SBATCH --image=zchandani731/nv_basf:v2
#SBATCH --module=cuda-mpich

export CUDAQ_MPI_COMM_LIB=${HOME}/distributed_interfaces/libcudaq_distributed_interface_mpi.so

srun --mpi=pmix shifter python3 test.py --cudaq-full-stack-trace