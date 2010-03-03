try:
	from pycuda.autoinit import device
	from pycuda.driver import device_attribute
	import pycuda.driver as cuda
	import pycuda.gpuarray as gpuarray
	from pycuda.compiler import SourceModule
except:
	pass

from mako.template import Template
import numpy

from globals import *
from transpose import Transpose

class Reduce:

	def __init__(self, precision, mempool):
		self._precision = precision
		self._mempool = mempool
		self._tr_scalar = Transpose(precision.scalar)
		self._tr_complex = Transpose(precision.complex)

		kernel_template = Template("""
		inline ${p.complex.name} operator+(${p.complex.name} a, ${p.complex.name} b)
		{ return ${p.complex.ctr}(a.x + b.x, a.y + b.y); }

		inline void operator+=(${p.complex.name}& a, const ${p.complex.name}& b)
		{ a.x += b.x; a.y += b.y; }

		extern "C" {
		%for typename in (p.scalar.name, p.complex.name):
		%for block_size in [2 ** x for x in xrange(log2(max_block_size) + 1)]:
			<%
				log2_warp_size = log2(warp_size)
				log2_block_size = log2(block_size)
				smem_size = block_size + (0 if block_size > warp_size else block_size / 2)
			%>

			__global__ void reduceKernel${block_size}${typename}(${typename}* output, const ${typename}* input)
			{
				__shared__ ${typename} shared_mem[${smem_size}];

				int tid = threadIdx.x;
				int bid = blockIdx.y * gridDim.x + blockIdx.x;

				// first reduction, after which the number of elements to reduce
				// equals to number of threads in block
				shared_mem[tid] = input[tid + 2 * bid * ${block_size}] +
					input[tid + 2 * bid * ${block_size} + ${block_size}];

				__syncthreads();

				// 'if(tid)'s will split execution only near the border of warps,
				// so they are not affecting performance (i.e, for each warp there
				// will be only one path of execution anyway)
				%for reduction_pow in xrange(log2_block_size - 1, log2_warp_size, -1):
					if(tid < ${2 ** reduction_pow})
						shared_mem[tid] += shared_mem[tid + ${2 ** reduction_pow}];
					__syncthreads();
				%endfor

				// The following code will be executed inside a single warp, so no
				// shared memory synchronization is necessary
				if (tid < ${warp_size}) {
				%for reduction_pow in xrange(log2_warp_size, -1, -1):
					%if block_size >= 2 ** (reduction_pow + 1):
						shared_mem[tid] += shared_mem[tid + ${2 ** reduction_pow}];
					%endif
				%endfor
				}

				if (tid == 0) output[bid] = shared_mem[0];
			}
		%endfor
		%endfor
		}
		""")

		self._max_block_size = device.get_attribute(device_attribute.MAX_BLOCK_DIM_X)
		warp_size =device.get_attribute(device_attribute.WARP_SIZE)

		kernel_src = kernel_template.render(p=self._precision,
			warp_size=warp_size, max_block_size=self._max_block_size, log2=log2)
		module = SourceModule(kernel_src, no_extern_c=True)

		self._scalar_kernels = {}
		for block_size in [2 ** x for x in xrange(log2(self._max_block_size) + 1)]:
			name = "reduceKernel" + str(block_size) + precision.scalar.name
			self._scalar_kernels[block_size] = FunctionWrapper(module, name, "PP", block_size=block_size)

		self._complex_kernels = {}
		for block_size in [2 ** x for x in xrange(log2(self._max_block_size) + 1)]:
			name = "reduceKernel" + str(block_size) + precision.complex.name
			self._complex_kernels[block_size] = FunctionWrapper(module, name, "PP", block_size=block_size)

	def __call__(self, array, final_length=1):

		length = array.size
		assert length >= final_length, "Array size cannot be less than final size"

		if array.dtype == self._precision.scalar.dtype:
			reduce_kernels = self._scalar_kernels
			itemsize = self._precision.scalar.nbytes
		else:
			reduce_kernels = self._complex_kernels
			itemsize = self._precision.complex.nbytes

		if length == final_length:
			res = gpuarray.GPUArray((length,), dtype=array.dtype, allocator=self._mempool)
			cuda.memcpy_dtod(res.gpudata, array.gpudata, length * itemsize)
			return res

		# we can reduce maximum block size * 2 times a pass
		max_reduce_power = self._max_block_size * 2

		data_in = array

		while length > final_length:

			reduce_power = max_reduce_power if length / final_length >= max_reduce_power else length / final_length

			data_out = gpuarray.GPUArray((data_in.size / reduce_power,), dtype=array.dtype,
					allocator=self._mempool)

			func = reduce_kernels[reduce_power / 2]
			func(length / 2, data_out.gpudata, data_in.gpudata)

			length /= reduce_power

			data_in = data_out

		if final_length == 1:
		# return reduction result
			result = numpy.array((1,), dtype=array.dtype)
			cuda.memcpy_dtoh(result, data_in.gpudata)
			return result[0]
		else:
			return data_in

	def sparse(self, array, final_length=1):
		if final_length == 1:
			return self(array)

		res = gpuaray.GPUArray(array.shape, dtype=array.dtype, allocator=self._mempool)
		reduce_power = array.size / final_length
		if array.dtype == self._precision.scalar.dtype:
			self._tr_scalar(res, array, final_length, reduce_power)
		else:
			self._tr_complex(res, array, final_length, reduce_power)

		return self(res, final_length=final_length)

class CPUReduce:

	def __call__(self, array, final_length=1):

		if final_length == 1:
			return numpy.sum(array)

		flat_array = array.ravel()
		res = numpy.empty(final_length)
		reduce_power = array.size / final_length

		for i in xrange(final_length):
			res[i] = numpy.sum(flat_array[i*reduce_power:(i+1)*reduce_power])

		return res

	def sparse(self, array, final_length=1):

		if final_length == 1:
			return self(array)

		reduce_power = array.size / final_length
		return self(numpy.transpose(array.reshape(final_length, reduce_power)), final_length=final_length)

def getReduce(gpu, precision, mempool):
	if gpu:
		return Reduce(precision, mempool)
	else:
		return CPUReduce()
