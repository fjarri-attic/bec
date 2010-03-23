"""
Ground state calculation classes
"""

import math
import copy
import numpy

from globals import *
from fft import createPlan
from meters import ParticleStatistics
from reduce import getReduce


class TFGroundState(PairedCalculation):
	"""
	Ground state, calculated using Thomas-Fermi approximation
	(kinetic energy == 0)
	"""

	def __init__(self, env):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._potentials = getPotentials(self._env)

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _cpu_create(self):
		res = self._env.allocate(self._env.constants.shape, self._env.precision.complex.dtype)

		for i in xrange(self._env.constants.nvx):
			for j in xrange(self._env.constants.nvy):
				for k in xrange(self._env.constants.nvz):
					e = self._env.constants.mu - self._potentials[k, j, i]
					res[k, j, i] = math.sqrt(max(e / self._env.constants.g11, 0))

		return res

	def _gpu__prepare(self):
		kernel_template = """
			// fill given buffer with ground state, obtained from Thomas-Fermi approximation
			__kernel void fillWithTFGroundState(__global ${p.complex.name} *data,
				read_only image3d_t potentials)
			{
				DEFINE_INDEXES;

				float potential = get_float_from_image(potentials, i, j, k);

				${p.scalar.name} e = (${p.scalar.name})${c.mu} - potential;
				if(e > 0)
					data[index] = ${p.complex.ctr}(sqrt(e / (${p.scalar.name})${c.g11}), 0);
				else
					data[index] = ${p.complex.ctr}(0, 0);
			}
		"""

		self._program = self._env.compileSource(kernel_template)
		self._func = FunctionWrapper(self._program.fillWithTFGroundState)

	def _gpu_create(self):
		res = self._env.allocate(self._env.constants.shape, self._env.precision.complex.dtype)
		self._func(self._env.queue, self._env.constants.shape, res, self._potentials)
		return res


class GPEGroundState(PairedCalculation):
	"""
	Calculates GPE ground state using split-step propagation in imaginary time.
	"""

	def __init__(self, env):
		PairedCalculation.__init__(self, env)
		self._env = env

		self._tf_gs = TFGroundState(env)
		self._plan = createPlan(env, env.constants.nvx, env.constants.nvy, env.constants.nvz)
		self._statistics = ParticleStatistics(env)

		self._potentials = getPotentials(self._env)
		self._kvectors = getKVectors(self._env)

		self._prepare()

		# condition for stopping propagation -
		# relative difference between state energies of two successive steps
		self._precision = 1e-6

	def _cpu__prepare(self):
		self._k_coeff = numpy.exp(self._kvectors * (-self._env.constants.dt_steady / 2))

	def _gpu__prepare(self):
		kernel_template = """
			float get_potential(int i, int j, int k)
			{
				float x = -${c.xmax} + i * ${c.dx};
				float y = -${c.ymax} + j * ${c.dy};
				float z = -${c.zmax} + k * ${c.dz};

				return (x * x + y * y + z * z /
					(${c.lambda_ * c.lambda_})) / 2;
			}
			__kernel void multiply(__global ${p.complex.name} *data, __global ${p.scalar.name} coeff)
			{
				DEFINE_INDEXES;
				data[index] = complex_mul_scalar(data[index], coeff);
			}

			// Propagates state vector in k-space for steady state calculation (i.e., in imaginary time)
			__kernel void propagateKSpaceImaginaryTime(__global ${p.complex.name} *data,
				read_only image3d_t kvectors)
			{
				DEFINE_INDEXES;

				${p.scalar.name} kvector = get_float_from_image(kvectors, i, j, k);

				${p.scalar.name} prop_coeff = native_exp(kvector *
					(${p.scalar.name})${-c.dt_steady / 2});
				${p.complex.name} temp = data[index];
				data[index] = complex_mul_scalar(temp, prop_coeff);
			}

			// Propagates state in x-space for steady state calculation
			__kernel void propagateXSpaceOneComponent(__global ${p.complex.name} *data,
				read_only image3d_t potentials)
			{
				DEFINE_INDEXES;

				${p.complex.name} a = data[index];

				//store initial x-space field
				${p.complex.name} a0 = a;

				${p.scalar.name} da;
				${p.scalar.name} V = get_float_from_image(potentials, i, j, k);

				//iterate to midpoint solution
				%for iter in range(c.itmax):
					//calculate midpoint log derivative and exponentiate
					da = exp((${p.scalar.name})${c.dt_steady / 2} *
						(-V - (${p.scalar.name})${c.g11} * squared_abs(a)));

					//propagate to midpoint using log derivative
					a = complex_mul_scalar(a0, da);
				%endfor

				//propagate to endpoint using log derivative
				data[index] = complex_mul_scalar(a, da);
			}
		"""

		self._program = self._env.compileSource(kernel_template)

		self._kpropagate_func = FunctionWrapper(self._program.propagateKSpaceImaginaryTime)
		self._xpropagate_func = FunctionWrapper(self._program.propagateXSpaceOneComponent)
		self._multiply_func = FunctionWrapper(self._program.multiply)

	def _cpu__kpropagate(self):
		self._gs *= self._k_coeff # k-space propagation

	def _gpu__kpropagate(self):
		self._kpropagate_func(self._env.queue, self._gs.shape, self._gs, self._kvectors)

	def _cpu__xpropagate(self):
		gs0 = self._gs.copy()
		for iter in xrange(self._env.constants.itmax):
			abs_gs = numpy.abs(self._gs)
			d_gs = numpy.exp((self._potentials + abs_gs * abs_gs * self._env.constants.g11) *
				(-self._env.constants.dt_steady / 2))
			self._gs = gs0 * d_gs
		self._gs *= d_gs

	def _gpu__xpropagate(self):
		self._xpropagate_func(self._env.queue, self._gs.shape, self._gs, self._potentials)

	def _cpu__renormalize(self, coeff):
		self._gs *= coeff

	def _gpu__renormalize(self, coeff):
		self._multiply_func(self._env.queue, self._gs.shape, self._gs, self._env.precision.scalar.dtype(coeff))

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
			self._renormalize(math.sqrt(self._env.constants.N / N))

			E = new_E
			new_E = stats.countEnergy(self._gs)
			plan.execute(self._gs, inverse=True) # FFT to k-space

		plan.execute(self._gs) # FFT to x-state

		print "Ground state calculation:" + \
			" N = " + str(stats.countParticles(self._gs, subtract_noise=False)) + \
			" E = " + str(stats.countEnergy(self._gs)) + \
			" mu = " + str(stats.countMu(self._gs))

		return self._gs
