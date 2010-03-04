"""
Ground state calculation classes
"""

import math
import copy
import numpy

try:
	import pycuda.driver as cuda
	from pycuda.compiler import SourceModule
	from pycuda import gpuarray
except:
	pass

from globals import *
from fft import createPlan
from meters import ParticleStatistics
from reduce import getReduce


class TFGroundState(PairedCalculation):
	"""
	Ground state, calculated using Thomas-Fermi approximation
	(kinetic energy == 0)
	"""

	def __init__(self, gpu, precision, constants, mempool):
		PairedCalculation.__init__(self, gpu)
		self._precision = precision
		self._constants = copy.deepcopy(constants)
		self._mempool = mempool

		self._prepare()

	def _cpu__prepare(self):
		self._potentials = fillPotentialsArray(self._precision, self._constants)

	def _cpu_create(self):
		res = numpy.empty(self._constants.shape, dtype=self._precision.complex.dtype)

		for i in xrange(self._constants.nvx):
			for j in xrange(self._constants.nvy):
				for k in xrange(self._constants.nvz):
					e = self._constants.mu - self._potentials[k, j, i]
					res[k, j, i] = math.sqrt(max(e / self._constants.g11, 0))

		return res

	def _gpu__prepare(self):
		kernel_template = """
			texture<${p.scalar.name}, 1> potentials;

			// fill given buffer with ground state, obtained from Thomas-Fermi approximation
			__global__ void fillWithTFGroundState(${p.complex.name} *data)
			{
				int index = GLOBAL_INDEX;

				${p.scalar.name} e = (${p.scalar.name})${c.mu} - tex1Dfetch(potentials, index);
				if(e > 0)
					data[index] = ${p.complex.ctr}(sqrt(e / (${p.scalar.name})${c.g11}), 0);
				else
					data[index] = ${p.complex.ctr}(0, 0);
			}
		"""

		self._module = compileSource(kernel_template, self._precision, self._constants)
		self._func = FunctionWrapper(self._module, "fillWithTFGroundState", "P")
		self._texref = self._module.get_texref("potentials")
		self._potentials_array = fillPotentialsTexture(self._precision, self._constants, self._texref)

	def _gpu_create(self):
		res = gpuarray.GPUArray(self._constants.shape, self._precision.complex.dtype, allocator=self._mempool)
		self._func(self._constants.cells, res.gpudata)
		return res


class GPEGroundState(PairedCalculation):
	"""
	Calculates GPE ground state using split-step propagation in imaginary time.
	"""

	def __init__(self, gpu, precision, constants, mempool):
		PairedCalculation.__init__(self, gpu)

		self._precision = precision
		self._constants = copy.deepcopy(constants)
		self._mempool = mempool
		self._gpu = gpu

		self._tf_gs = TFGroundState(gpu, precision, constants, mempool)
		self._plan = createPlan(gpu, constants.nvx, constants.nvy, constants.nvz, precision)
		self._statistics = ParticleStatistics(gpu, precision, constants, mempool)

		self._prepare()

		# condition for stopping propagation -
		# relative difference between state energies of two successive steps
		self._precision = 1e-6

	def _cpu__prepare(self):
		self._potentials = fillPotentialsArray(self._precision, self._constants)
		self._kvectors = fillKVectorsArray(self._precision, self._constants)

	def _cpu__kpropagate(self):
		self._gs *= numpy.exp(self._kvectors * (-self._constants.dt_steady / 2)) # k-space propagation

	def _gpu__kpropagate(self):
		self._kpropagate_func(self._constants.cells, self._gs.gpudata)

	def _gpu__xpropagate(self):
		self._xpropagate_func(self._constants.cells, self._gs.gpudata)

	def _cpu__xpropagate(self):
		gs0 = self._gs.copy()
		for iter in xrange(self._constants.itmax):
			abs_gs = numpy.abs(self._gs)
			d_gs = numpy.exp((self._potentials + abs_gs * abs_gs * self._constants.g11) *
				(-self._constants.dt_steady / 2))
			self._gs = gs0 * d_gs
		self._gs *= d_gs

	def _cpu__renormalize(self, coeff):
		self._gs *= coeff

	def _gpu__renormalize(self, coeff):
		self._multiply_func(self._constants.cells, self._gs.gpudata, coeff)

	def create(self):

		self._gs = self._tf_gs.create()
		plan = self._plan
		stats = self._statistics

		E = 0
		new_E = stats.countEnergy(self._gs)

		plan.execute(self._gs, inverse=True) # FFT to k-space

		while abs(E - new_E) / E > self._precision:

			# propagation

			self._kpropagate()
			plan.execute(self._gs) # FFT to x-space
			self._xpropagate()
			plan.execute(self._gs, inverse=True) # FFT to k-space
			self._kpropagate()

			# normalization

			plan.execute(self._gs) # FFT to x-space

			# renormalize
			N = stats.countParticles(self._gs, subtract_noise=False)
			self._renormalize(math.sqrt(self._constants.N / N))

			E = new_E
			new_E = stats.countEnergy(self._gs)
			plan.execute(self._gs, inverse=True) # FFT to k-space

		plan.execute(self._gs) # FFT to x-state

		print "Ground state calculation:" + \
			" N = " + str(stats.countParticles(self._gs, subtract_noise=False)) + \
			" E = " + str(stats.countEnergy(self._gs)) + \
			" mu = " + str(stats.countMu(self._gs))

		return self._gs

	def _gpu__prepare(self):
		kernel_template = """
			texture<${p.scalar.name}, 1> potentials;
			texture<${p.scalar.name}, 1> kvectors;

			__global__ void multiply(${p.complex.name} *data, ${p.scalar.name} coeff)
			{
				int index = GLOBAL_INDEX;
				data[index] = data[index] * coeff;
			}

			// Propagates state vector in k-space for steady state calculation (i.e., in imaginary time)
			__global__ void propagateKSpaceImaginaryTime(${p.complex.name} *data)
			{
				int index = GLOBAL_INDEX;
				${p.scalar.name} prop_coeff = exp(tex1Dfetch(kvectors, index) *
					${p.scalar.name}(${-c.dt_steady / 2}));
				${p.complex.name} temp = data[index];
				data[index] = temp * prop_coeff;
			}

			// Propagates state in x-space for steady state calculation
			__global__ void propagateXSpaceOneComponent(${p.complex.name} *data)
			{
				int index = GLOBAL_INDEX;

				${p.complex.name} a = data[index];

				//store initial x-space field
				${p.complex.name} a0 = a;

				${p.scalar.name} da;
				${p.scalar.name} V = tex1Dfetch(potentials, index);

				//iterate to midpoint solution
				%for iter in range(c.itmax):
					//calculate midpoint log derivative and exponentiate
					da = exp(${p.scalar.name}(${c.dt_steady / 2}) *
						(-V - ${p.scalar.name}(${c.g11}) * squared_abs(a)));

					//propagate to midpoint using log derivative
					a = a0 * da;
				%endfor

				//propagate to endpoint using log derivative
				data[index] = a * da;
			}
		"""

		self._module = compileSource(kernel_template, self._precision, self._constants)

		self._kpropagate_func = FunctionWrapper(self._module, "propagateKSpaceImaginaryTime", "P")
		self._xpropagate_func = FunctionWrapper(self._module, "propagateXSpaceOneComponent", "P")
		self._multiply_func = FunctionWrapper(self._module, "multiply", "P" + self._precision.scalar.ctype)
		self._potentials_texref = self._module.get_texref("potentials")
		self._kvectors_texref = self._module.get_texref("kvectors")

		self._potentials_array = fillPotentialsTexture(self._precision, self._constants, self._potentials_texref)
		self._kvectors_array = fillKVectorsTexture(self._precision, self._constants, self._kvectors_texref)
