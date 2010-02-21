import math
import copy
from mako.template import Template
import numpy

try:
	import pycuda.driver as cuda
	from pycuda.compiler import SourceModule
	from pycuda import gpuarray
except:
	pass

from globals import *

class GroundState:

	def __init__(self, gpu, precision, constants, mempool):
		self._precision = precision
		self._constants = copy.deepcopy(constants)
		self._mempool = mempool
		self._gpu = gpu

		if self._gpu:
			self._gpuPrepare()
		else:
			self._cpuPrepare()

	def create(self):
		if self._gpu:
			return self.gpuCreate()
		else:
			return self.cpuCreate()

	def _cpuPrepare(self):
		self._potentials = fillPotentialsArray(self._precision, self._constants)

	def cpuCreate(self):
		res = numpy.empty(self._constants.shape, dtype=self._precision.complex.dtype)

		for i in xrange(self._constants.nvx):
			for j in xrange(self._constants.nvy):
				for k in xrange(self._constants.nvz):
					e = self._constants.mu - self._potentials[k, j, i]
					res[k, j, i] = math.sqrt(max(e / self._constants.g11, 0))

		return res

	def _gpuPrepare(self):
		kernel_template = Template("""
			texture<${precision.scalar.name}, 1> potentials;

			// fill given buffer with ground state, obtained from Thomas-Fermi approximation
			__global__ void fillWithTFGroundState(${precision.complex.name} *data)
			{
				int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

				${precision.scalar.name} e = (${precision.scalar.name})${constants.mu} - tex1Dfetch(potentials, index);
				if(e > 0)
					data[index] = ${precision.complex.ctr}(sqrt(e / (${precision.scalar.name})${constants.g11}), 0);
				else
					data[index] = ${precision.complex.ctr}(0, 0);
			}
		""")

		kernel_src = kernel_template.render(
			precision=self._precision,
			constants=self._constants)

		self._module = SourceModule(kernel_src)

		self._func = self._module.get_function("fillWithTFGroundState")
		self._texref = self._module.get_texref("potentials")
		fillPotentialsTexture(self._precision, self._constants, self._texref)

		block, self._grid = getExecutionParameters(self._func, self._constants.cells)
		self._func.prepare("P", block=block)

	def gpuCreate(self):
		res = gpuarray.GPUArray(self._constants.shape, self._precision.complex.dtype, allocator=self._mempool)
		self._func.prepared_call(self._grid, res.gpudata)
		return res

def fillPotentialsArray(precision, constants):
	potentials = numpy.empty(constants.shape, dtype=precision.scalar.dtype)

	for i in xrange(constants.nvx):
		for j in xrange(constants.nvy):
			for k in xrange(constants.nvz):
				x = -constants.xmax + i * constants.dx
				y = -constants.ymax + j * constants.dy
				z = -constants.zmax + k * constants.dz

				potentials[k, j, i] = (x * x + y * y + z * z / (constants.lambda_ * constants.lambda_)) / 2

	return potentials

def fillPotentialsTexture(precision, constants, texref):
	potentials = fillPotentialsArray(precision, constants)
	cuda.matrix_to_texref(potentials.reshape(1, constants.cells), texref, order="C")
	texref.set_filter_mode(cuda.filter_mode.POINT)
	texref.set_address_mode(0, cuda.address_mode.CLAMP)

def fillKVectorsTexture(constants, texref):

	kvalue = lambda i, dk, N: dk * (i - N) if 2 * i > N else dk * i

	vectors = numpy.empty(constants.cells, dtype=precision.scalar.dtype)
	for index in xrange(constants.cells):
		k = index >> (constants.nvx_pow + constants.nvy_pow)
		k_shift = (k << (constants.nvx_pow + constants.nvy_pow))
		j = (index - k_shift) >> constants.nvx_pow
		i = (index - k_shift) - (j << constants.nvx_pow)

		kx = kvalue(i, params.dkx, params.nvx)
		ky = kvalue(j, params.dky, params.nvy)
		kz = kvalue(k, params.dkz, params.nvz)

		vectors[index] = (kx * kx + ky * ky + kz * kz) / 2

	cuda.matrix_to_texref(vectors, texref, order="F")
	texref.set_filter_mode(cuda.filter_mode.POINT)
	texref.set_address_mode(0, cuda.address_mode.CLAMP)
