"""
Text the ansatz
"""
import numpy as np

from src.vqe_cudaq_qnp import VQE
from src.vqe_cudaq_qnp import get_cudaq_hamiltonian
import pickle
from itertools import product
import time
import pandas as pd
import os
from datetime import datetime

n_qubits = 4
num_active_electrons = 2
spin = 0
start_t = time.time()
vqe = VQE(n_qubits=n_qubits,
          num_active_electrons=num_active_electrons,
          spin=spin,
          options={})

kernel, thetas = vqe.layers()
