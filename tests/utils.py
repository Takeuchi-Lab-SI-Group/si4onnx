import os
import time
import torch
import numpy as np
from contextlib import contextmanager

def set_seed(seed):
  np.random.seed(0)
  torch.manual_seed(0)

def set_thread():
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["OPENBLAS_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["VECLIB_NUM_THREADS"] = "1"
  os.environ["NUMEXPR_NUM_THREADS"] = "1"
  torch.set_num_threads(1)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')