from mako.template import Template
import numpy
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import math

from config import precision
from globals import *

def createTFGroundStateCPU(constants):
	res = numpy.empty((constants.nvx, constants.nvy, constants.nvz), dtype=precision.complex.dtype)

	for i in xrange(constants.nvx):
		for j in xrange(constants.nvy):
			for k in xrange(constants.nvz):
				x = -constants.xmax + constants.dx * i
				y = -constants.ymax + constants.dy * j
				z = -constants.zmax + constants.dz * k

				potential = (x * x + y * y + z * z / (constants.lambda_ * constants.lambda_)) / 2
				e = constants.mu - potential
				res[i, j, k] = math.sqrt(max(e / constants.g11, 0))

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

def createTFGroundStateGPU(precision, constants):
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
		precision=precision,
		constants=constants)
	print kernel_src
	module = SourceModule(kernel_src)
	func = module.get_function("fillWithTFGroundState")

	res = mempool.allocate(constants.cells * precision.complex.nbytes)

	block, grid = getExecutionParameters(func, constants.cells)
	print(block, grid)
	func.prepare("P", block=block)
	func.prepared_call(grid, res)
	return res
