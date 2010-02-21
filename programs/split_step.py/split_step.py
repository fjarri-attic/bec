import math
import matplotlib
import numpy

try:
	import pycuda.autoinit
	import pycuda.driver as cuda
	mempool = GPUPool()
except:
	pass

from globals import GPUPool
from config import Model, precision
from constants import Constants
from ground_state import GroundState

constants = Constants(Model)
mempool = GPUPool(stub=True)
gpu = True

gs = GroundState(gpu, precision, constants, mempool)
gs_gpu = gs.create()
gs_gpu_h = gs_gpu.get()

print numpy.sum(numpy.abs(gs_gpu_h))

