try:
	from pycuda.autoinit import device
	from pycuda.driver import device_attribute
	import pycuda.driver as cuda
	from pycuda.tools import DeviceMemoryPool
	from pycuda.compiler import SourceModule
	import pycuda.gpuarray as gpuarray
	pycuda_available = True
except:
	pycuda_available = False

from mako.template import Template
import numpy


class GPUPool:

	def __init__(self, stub=False):
		if not pycuda_available:
			self.allocate = None
			return

		if stub:
			self.allocate = cuda.mem_alloc
		else:
			self._pool = DeviceMemoryPool()
			self.allocate = self._pool.allocate

	def __call__(self, size):
		return self.allocate(size)


class PairedCalculation:
	def __init__(self, gpu):
		prefix = "_gpu_" if gpu else "_cpu_"

		for attr in dir(self):
			if attr.startswith(prefix):
				name = attr[len(prefix):]
				self.__dict__[name] = getattr(self, attr)


class FunctionWrapper:

	def __init__(self, module, name, arg_list, block_size=None):
		self._module = module
		self._name = name
		self._arg_list = arg_list
		self._prepared_refs = {}
		self._block_size = block_size

	def _prepare(self, elements):
		func_ref = self._module.get_function(self._name)
		block, grid = getExecutionParameters(func_ref, elements, block_size=self._block_size)
		func_ref.prepare(self._arg_list, block)
		self._prepared_refs[elements] = (func_ref, grid)

	def __call__(self, elements, *args):
		if elements not in self._prepared_refs:
			self._prepare(elements)

		func_ref, grid = self._prepared_refs[elements]
		func_ref.prepared_call(grid, *args)


KERNEL_DEFINES = Template("""
	inline ${p.complex.name} operator+(${p.complex.name} a, ${p.complex.name} b)
	{ return ${p.complex.ctr}(a.x + b.x, a.y + b.y); }
	inline ${p.complex.name} operator-(${p.complex.name} a, ${p.complex.name} b)
	{ return ${p.complex.ctr}(a.x - b.x, a.y - b.y); }
	inline ${p.complex.name} operator*(${p.complex.name} a, ${p.scalar.name}  b)
	{ return ${p.complex.ctr}(b * a.x, b * a.y); }
	inline ${p.complex.name} operator*(${p.complex.name} a, ${p.complex.name} b)
	{ return ${p.complex.ctr}(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
	inline __device__ ${p.scalar.name} squared_abs(${p.complex.name} a)
	{ return a.x * a.x + a.y * a.y; }
	inline void operator+=(${p.complex.name}& a, const ${p.complex.name}& b)
	{ a.x += b.x; a.y += b.y; }
	inline __device__ ${p.complex.name} cexp(${p.complex.name} a)
	{
		${p.scalar.name} module = exp(a.x);
		return ${p.complex.ctr}(module * cos(a.y), module * sin(a.y));
	}

	#define GLOBAL_INDEX (threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x))
""")


def compileSource(source, precision, constants, **kwds):
	defines = KERNEL_DEFINES.render(p=precision)
	kernel_src = source.render(p=precision, c=constants, **kwds)
	return SourceModule(defines + "extern \"C\" {\n" + kernel_src + "\n}", no_extern_c=True)

def log2(x):
	"""Auxiliary function, calculating binary logarithm for integer"""
	pows = [1]
	while x > 2 ** pows[-1]:
		pows.append(pows[-1] * 2)

	res = 0
	for pow in reversed(pows):
		if x >= (2 ** pow):
			x >>= pow
			res += pow
	return res

def getExecutionParameters(func, elements, block_size=None):
	max_block_size = device.get_attribute(device_attribute.MAX_BLOCK_DIM_X)
	max_registers = device.get_attribute(device_attribute.MAX_REGISTERS_PER_BLOCK)

	if block_size is None:
		block_size = min(max_block_size, 2 ** log2(max_registers / func.num_regs))
		assert block_size > 0, "Too much registers used by kernel"
	else:
		assert block_size <= max_block_size, "Fixed block size is too big"
		assert block_size * func.num_regs <= max_registers, "Not enough registers for fixed block size"

	max_grid_x = device.get_attribute(device_attribute.MAX_GRID_DIM_X)
	max_grid_y = device.get_attribute(device_attribute.MAX_GRID_DIM_Y)
	blocks_num_x = min(max_grid_x, elements / block_size)
	blocks_num_y = 1 if blocks_num_x <= elements else elements / blocks_num_x

	assert blocks_num_y <= max_grid_y, "Insufficient grid size to handle all the elements"

	return (block_size, 1, 1), (blocks_num_x, blocks_num_y)

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
	array = gpuarray.to_gpu(potentials)
	texref.set_address(array.gpudata, array.size * potentials.itemsize, allow_offset=False)
	#cuda.matrix_to_texref(potentials.reshape(1, constants.cells), texref, order="C")
	texref.set_format(cuda.array_format.FLOAT, 1)
	texref.set_filter_mode(cuda.filter_mode.POINT)
	texref.set_address_mode(0, cuda.address_mode.CLAMP)
	return array

def fillKVectorsArray(precision, constants):

	kvalue = lambda i, dk, N: dk * (i - N) if 2 * i > N else dk * i
	kvectors = numpy.empty(constants.shape, dtype=precision.scalar.dtype)

	for i in xrange(constants.nvx):
		for j in xrange(constants.nvy):
			for k in xrange(constants.nvz):

				kx = kvalue(i, constants.dkx, constants.nvx)
				ky = kvalue(j, constants.dky, constants.nvy)
				kz = kvalue(k, constants.dkz, constants.nvz)

				kvectors[k, j, i] = (kx * kx + ky * ky + kz * kz) / 2

	return kvectors

def fillKVectorsTexture(precision, constants, texref):
	kvectors = fillKVectorsArray(precision, constants)
	array = gpuarray.to_gpu(kvectors)
	texref.set_address(array.gpudata, array.size * kvectors.itemsize, allow_offset=False)
	#cuda.matrix_to_texref(kvectors.reshape(1, constants.cells), texref, order="C")
	texref.set_format(cuda.array_format.FLOAT, 1)
	texref.set_filter_mode(cuda.filter_mode.POINT)
	texref.set_address_mode(0, cuda.address_mode.CLAMP)
	return array
