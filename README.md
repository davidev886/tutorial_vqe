### shifter installation

```
shifter --image=docker:nvcr.io/nvidia/nightly/cuda-quantum:latest --module=cuda-mpich /bin/bash 

cp -r /opt/nvidia/cudaq/distributed_interfaces/ .

pip install pyscf openfermion openfermionpyscf

pip install cupy-cuda12x

install requirements from tutorial_vqe

exit shifter

export MPI_PATH=/opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1
source distributed_interfaces/activate_custom_mpi.sh


echo $CUDAQ_MPI_COMM_LIB

shifter --image=docker:nvcr.io/nvidia/nightly/cuda-quantum:latest --module=cuda-mpich /bin/bash

cp /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.11.8.89 ~/libcudart.so

exit
```

### Running cudaq VQE only:

```
salloc -N 1 --gpus-per-task=4 --ntasks-per-node=1 -t 30 --qos=interactive -A m4465 -C gpu --image=docker:nvcr.io/nvidia/nightly/cuda-quantum:latest --module=cuda-mpich
export LD_LIBRARY_PATH=$HOME:$LD_LIBRARY_PATH
export CUDAQ_MPI_COMM_LIB=${HOME}/distributed_interfaces/libcudaq_distributed_interface_mpi.so
srun --mpi=pmix shifter bash -l launch.sh test_vqe.py 
```

### Running ipie only:

```
salloc -N 1 --gpus-per-task=4 --ntasks-per-node=1 -t 30 --qos=interactive -A m4465 -C gpu
module load python
conda activate gpu-ipie
 
export LD_LIBRARY_PATH=$HOME:$LD_LIBRARY_PATH
export CUDAQ_MPI_COMM_LIB=${HOME}/distributed_interfaces/libcudaq_distributed_interface_mpi.so
srun --mpi=pmix shifter bash -l launch.sh test_vqe.py 
```

### Running CPUS only:

```
salloc --qos=interactive -C cpu --time=00:30:00 --nodes=1 -N 1 -A m4465 --image=docker:nvcr.io/nvidia/nightly/cuda-quantum:latest

export LD_LIBRARY_PATH=$HOME:$LD_LIBRARY_PATH
export CUDAQ_MPI_COMM_LIB=${HOME}/distributed_interfaces/libcudaq_distributed_interface_mpi.so
srun --mpi=pmix shifter bash -l launch.sh test_ipie.py
```
or
```
shifter python test_ipie.py 
```
