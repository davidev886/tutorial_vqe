#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpu-bind=none
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH -A m4642
#SBATCH -C gpu
#SBATCH --image=nersc/sc_cuda_quantum:24.10
#SBATCH --module=cuda-mpich

export LD_LIBRARY_PATH=$HOME:$LD_LIBRARY_PATH
export CUDAQ_MPI_COMM_LIB=${HOME}/distributed_interfaces/libcudaq_distributed_interface_mpi.so

srun --mpi=pmix shifter bash -l launch.sh complete_workflow_10q.py
