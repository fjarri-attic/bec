"""
Auxiliary functions and classes.
"""

try:
	import pyopencl as cl
except:
	pass

from mako.template import Template
import numpy


class Buffer(cl.Buffer):

	def __init__(self, context, shape, dtype):
		self.size = 1
		for dim in shape:
			self.size *= dim

		if isinstance(dtype, numpy.dtype):
			self.itemsize = dtype.itemsize
		else:
			self.itemsize = dtype().nbytes

		self.nbytes = self.itemsize * self.size

		cl.Buffer.__init__(self, context, cl.mem_flags.READ_WRITE, size=self.nbytes)
		self.shape = shape
		self.dtype = dtype


class Environment:

	def __init__(self, gpu, precision, constants):

		self.gpu = gpu
		self.precision = precision
		self.constants = constants

		if gpu:
			devices = []
			for platform in cl.get_platforms():
				devices.extend(platform.get_devices(device_type=cl.device_type.GPU))

			self.device = devices[0]
			self.context = cl.Context(devices=[self.device])
			self.queue = cl.CommandQueue(self.context)

		if gpu:
			self.allocate = self.__gpu_allocate
		else:
			self.allocate = self.__cpu_allocate

	def __cpu_allocate(self, shape, dtype):
		return numpy.empty(shape, dtype=dtype)

	def __gpu_allocate(self, shape, dtype):
		return Buffer(self.context, shape, dtype)

	def synchronize(self):
		if self.gpu:
			self.queue.finish()

	def __str__(self):
		if self.gpu:
			return "gpu"
		else:
			return "cpu"

	def compileSource(self, source, **kwds):
		"""
		Adds helper functions and defines to given source, renders it,
		compiles and returns Cuda module.
		"""

		kernel_defines = Template("""
			//#define complex_mul_scalar(a, b) ${p.complex.ctr}((a).x * (b), (a).y * (b))
			//#define complex_mul(a, b) ${p.complex.ctr}(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y))
			//#define squared_abs(a) ((a).x * (a).x + (a).y * (a).y)

			${p.complex.name} complex_mul_scalar(${p.complex.name} a, ${p.scalar.name} b)
			{
				return ${p.complex.ctr}(a.x * b, a.y * b);
			}

			${p.complex.name} complex_mul(${p.complex.name} a, ${p.complex.name} b)
			{
				return ${p.complex.ctr}(mad(-a.y, b.y, a.x * b.x), mad(a.y, b.x, a.x * b.y));
			}

			${p.scalar.name} squared_abs(${p.complex.name} a)
			{
				return a.x * a.x + a.y * a.y;
			}

			${p.complex.name} cexp(${p.complex.name} a)
			{
				${p.scalar.name} module = exp(a.x);
				${p.scalar.name} angle = a.y;
				return ${p.complex.ctr}(module * native_cos(angle), module * native_sin(angle));
			}

			float get_float_from_image(read_only image3d_t image, int i, int j, int k)
			{
				uint4 image_data = read_imageui(image,
					CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP | CLK_NORMALIZED_COORDS_FALSE,
					(int4)(i, j, k, 0));

				return *((float*)&image_data);
			}

			#define DEFINE_INDEXES int i = get_global_id(0), j = get_global_id(1), k = get_global_id(2), index = (k << ${c.nvx_pow + c.nvy_pow}) + (j << ${c.nvx_pow}) + i
		""")

		defines = kernel_defines.render(p=self.precision, c=self.constants)
		kernel_src = Template(source).render(p=self.precision, c=self.constants, **kwds)
		return cl.Program(self.context, defines + kernel_src).build()


class PairedCalculation:
	"""
	Base class for paired GPU/CPU calculations.
	Depending on initializing parameter, it will make visible either _gpu_
	or _cpu_ methods.
	"""

	def __init__(self, env):
		if env.gpu:
			prefix = "_gpu_"
		else:
			prefix = "_cpu_"

		for attr in dir(self):
			if attr.startswith(prefix):
				name = attr[len(prefix):]
				self.__dict__[name] = getattr(self, attr)


class FunctionWrapper:
	"""
	Wrapper for elementwise CL kernel. Caches prepared functions for
	calls with same element number.
	"""

	def __init__(self, kernel):
		self._kernel = kernel
		self._prepared = {}

		self._device = kernel.program.devices[0]
		self._kernel_max_wg_size = self._kernel.get_work_group_info(
			cl.kernel_work_group_info.WORK_GROUP_SIZE, self._device)
		self._device_max_wg_sizes = self._device.get_info(
			cl.device_info.MAX_WORK_ITEM_SIZES)

	def _prepare(self, shape):
		reversed_shape = tuple(reversed(shape))
		local_size = []
		wg_size = 1
		for i, e in enumerate(reversed_shape):
			local_dim = min(self._device_max_wg_sizes[i], self._kernel_max_wg_size / wg_size, e)
			local_size.append(local_dim)
			wg_size *= local_dim

		self._prepared[shape] = tuple(local_size)

	def __call__(self, queue, shape, *args):
		#if shape not in self._prepared:
		#	self._prepare(shape)

		#local_size = self._prepared[shape]
		shape = tuple(reversed(shape))
		self._kernel(queue, shape, *args)


def log2(x):
	"""Calculates binary logarithm for integer"""
	pows = [1]
	while x > 2 ** pows[-1]:
		pows.append(pows[-1] * 2)

	res = 0
	for pow in reversed(pows):
		if x >= (2 ** pow):
			x >>= pow
			res += pow
	return res

def getPotentials(env):
	"""Returns array with values of external potential energy."""

	potentials = numpy.empty(env.constants.shape, dtype=env.precision.scalar.dtype)

	for i in xrange(env.constants.nvx):
		for j in xrange(env.constants.nvy):
			for k in xrange(env.constants.nvz):
				x = -env.constants.xmax + i * env.constants.dx
				y = -env.constants.ymax + j * env.constants.dy
				z = -env.constants.zmax + k * env.constants.dz

				potentials[k, j, i] = (x * x + y * y + z * z /
					(env.constants.lambda_ * env.constants.lambda_)) / 2

	if not env.gpu:
		return potentials

	return cl.Image(env.context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
		cl.ImageFormat(cl.channel_order.R, cl.channel_type.UNSIGNED_INT32),
		shape=tuple(reversed(env.constants.shape)), hostbuf=potentials)

def getKVectors(env):
	"""Returns array with values of k-space vectors."""

	def kvalue(i, dk, N):
		if 2 * i > N:
			return dk * (i - N)
		else:
			return dk * i

	kvectors = numpy.empty(env.constants.shape, dtype=env.precision.scalar.dtype)

	for i in xrange(env.constants.nvx):
		for j in xrange(env.constants.nvy):
			for k in xrange(env.constants.nvz):

				kx = kvalue(i, env.constants.dkx, env.constants.nvx)
				ky = kvalue(j, env.constants.dky, env.constants.nvy)
				kz = kvalue(k, env.constants.dkz, env.constants.nvz)

				kvectors[k, j, i] = (kx * kx + ky * ky + kz * kz) / 2

	if not env.gpu:
		return kvectors

	return cl.Image(env.context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
		cl.ImageFormat(cl.channel_order.R, cl.channel_type.UNSIGNED_INT32),
		shape=tuple(reversed(env.constants.shape)), hostbuf=kvectors)
