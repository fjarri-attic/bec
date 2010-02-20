import math
import matplotlib
import numpy
import pycuda.autoinit
from pycuda.tools import DeviceMemoryPool
import pycuda.driver as cuda

from config import Model, precision
from constants import *
from steady import *

constants = Constants(Model)
#mempool = DeviceMemoryPool()

class NoPool:
	def allocate(self, size):
		return cuda.mem_alloc(size)
mempool = NoPool()
gs = GroundState(precision, constants, mempool)

gs_cpu = gs.create(gpu=False)
gs_gpu = gs.create(gpu=True)

gs_gpu_h = numpy.empty(gs_cpu.shape, dtype=precision.complex.dtype)
cuda.memcpy_dtoh(gs_gpu_h, gs_gpu)


def sum(a):
	abs_a = numpy.abs(a)
	return numpy.sum(abs_a * abs_a)

def measure(a, modifier):
	abs_a = numpy.abs(a)
	return numpy.sum(abs_a * abs_a - modifier)

states = []
for i in range(constants.ensembles):
	psi = createTFGroundStateCPU(constants)
	dev = 0.5
	noise = numpy.random.normal(scale=dev, size=psi.shape) + 1j * numpy.random.normal(scale=dev, size=psi.shape)
	noised_psi = psi + noise / math.sqrt(constants.dV)
	print sum(psi) * constants.dV, sum(noised_psi) * constants.dV, measure(noised_psi, 0.5 / constants.dV) * constants.dV
	states.append(noised_psi)

averages = numpy.zeros(constants.shape, dtype=states[0].dtype)
for i in range(constants.ensembles):
	averages += states[i]

averages /= constants.ensembles

squares = numpy.zeros(constants.shape, dtype=states[0].dtype)
for i in range(constants.ensembles):
	tmp = states[i] - averages
	squares += numpy.real(tmp) * numpy.real(tmp) + 1j * numpy.imag(tmp) * numpy.imag(tmp)
squares /= constants.ensembles

print numpy.sum(squares) / squares.size * constants.dV
