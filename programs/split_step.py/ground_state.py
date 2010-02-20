import math
import copy
from mako.template import Template
import numpy

try:
	import pycuda.driver as cuda
	from pycuda.compiler import SourceModule
except:
	pass

from globals import *

class GroundState:

	def __init__(self, precision, constants, mempool):
		self._precision = precision
		self._constants = copy.deepcopy(constants)
		self._mempool = mempool

	def create(self, gpu=True):
		if gpu:
			return self.gpuCreate()
		else:
			return self.cpuCreate()

	def cpuCreate(self):
		res = numpy.empty(self._constants.shape, dtype=self._precision.complex.dtype)

		for i in xrange(self._constants.nvx):
			for j in xrange(self._constants.nvy):
				for k in xrange(self._constants.nvz):
					x = -self._constants.xmax + self._constants.dx * i
					y = -self._constants.ymax + self._constants.dy * j
					z = -self._constants.zmax + self._constants.dz * k

					potential = (x * x + y * y + z * z / (self._constants.lambda_ * self._constants.lambda_)) / 2
					e = self._constants.mu - potential
					res[k, j, i] = math.sqrt(max(e / self._constants.g11, 0))

		return res

	def gpuCreate(self):
		kernel_template = Template("""
			// Returns external potential energy for given lattice node
			__device__ __inline__ ${precision.scalar.name} potential(int index)
			{
				int k = index >> (${constants.nvx_pow} + ${constants.nvy_pow});
				index -= (k << (${constants.nvx_pow} + ${constants.nvy_pow}));
				int j = index >> ${constants.nvx_pow};
				int i = index - (j << ${constants.nvx_pow});

				${precision.scalar.name} x = -${constants.xmax} + ${constants.dx} * i;
				${precision.scalar.name} y = -${constants.ymax} + ${constants.dy} * j;
				${precision.scalar.name} z = -${constants.zmax} + ${constants.dz} * k;

				return (x * x + y * y + z * z / (${constants.lambda_} * ${constants.lambda_})) / 2;
			}

			// fill given buffer with ground state, obtained from Thomas-Fermi approximation
			__global__ void fillWithTFGroundState(${precision.complex.name} *data)
			{
				int index = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);

				${precision.scalar.name} e = ${constants.mu} - potential(index);
				if(e > 0)
					data[index] = ${precision.complex.ctr}(sqrt(e / ${constants.g11}), 0);
				else
					data[index] = ${precision.complex.ctr}(0, 0);
			}
		""")

		kernel_src = kernel_template.render(
			precision=self._precision,
			constants=self._constants)

		module = SourceModule(kernel_src)
		func = module.get_function("fillWithTFGroundState")
		block, grid = getExecutionParameters(func, self._constants.cells)

		res = self._mempool.allocate(self._constants.cells * self._precision.complex.nbytes)

		func.prepare("P", block=block)
		func.prepared_call(grid, res)

		return res


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

	cuda.matrix_to_texref(vector, texref, order="F")
	texref.set_filter_mode(cuda.filter_mode.POINT)
	texref.set_address_mode(0, cuda.address_mode.CLAMP)
