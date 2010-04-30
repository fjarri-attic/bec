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

	def _cpu_create(self, component=1):
		res = self._env.allocate(self._env.constants.shape, self._env.precision.complex.dtype)

		if component == 1:
			g = self._env.constants.g11
			mu = self._env.constants.mu
		else:
			g = self._env.constants.g22
			mu = self._env.constants.mu2

		for i in xrange(self._env.constants.nvx):
			for j in xrange(self._env.constants.nvy):
				for k in xrange(self._env.constants.nvz):
					e = mu - self._potentials[k, j, i]
					res[k, j, i] = math.sqrt(max(e / g, 0))

		return res

	def _gpu__prepare(self):
		kernel_template = """
			// fill given buffer with ground state, obtained from Thomas-Fermi approximation
			__kernel void fillWithTFGroundState(__global ${p.complex.name} *data,
				read_only image3d_t potentials, ${p.scalar.name} mu,
				${p.scalar.name} g)
			{
				DEFINE_INDEXES;

				float potential = get_float_from_image(potentials, i, j, k);

				${p.scalar.name} e = mu - potential;
				if(e > 0)
					data[index] = ${p.complex.ctr}(sqrt(e / g), 0);
				else
					data[index] = ${p.complex.ctr}(0, 0);
			}
		"""

		self._program = self._env.compileSource(kernel_template)
		self._func = FunctionWrapper(self._program.fillWithTFGroundState)

	def _gpu_create(self, component=1):
		if component == 1:
			g = self._env.constants.g11
			mu = self._env.constants.mu
		else:
			g = self._env.constants.g22
			mu = self._env.constants.mu2

		res = self._env.allocate(self._env.constants.shape, self._env.precision.complex.dtype)
		self._func(self._env.queue, self._env.constants.shape, res, self._potentials,
			self._env.precision.scalar.cast(mu), self._env.precision.scalar.cast(g))
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
			${p.scalar.name} get_potential(int i, int j, int k)
			{
				${p.scalar.name} x = -${c.xmax} + i * ${c.dx};
				${p.scalar.name} y = -${c.ymax} + j * ${c.dy};
				${p.scalar.name} z = -${c.zmax} + k * ${c.dz};

				return (x * x + y * y + z * z /
					(${c.lambda_ * c.lambda_})) / 2;
			}

			${p.scalar.name} get_kvector(int i, int j, int k)
			{
				${p.scalar.name} kx = (2 * i > ${c.nvx}) ? ((${p.scalar.name})${c.dkx} * (i - ${c.nvx})) : ((${p.scalar.name})${c.dkx} * i);
				${p.scalar.name} ky = (2 * j > ${c.nvy}) ? ((${p.scalar.name})${c.dky} * (j - ${c.nvy})) : ((${p.scalar.name})${c.dky} * j);
				${p.scalar.name} kz = (2 * k > ${c.nvz}) ? ((${p.scalar.name})${c.dkz} * (k - ${c.nvz})) : ((${p.scalar.name})${c.dkz} * k);

				return (kx * kx + ky * ky + kz * kz) / 2;
			}

			__kernel void multiply(__global ${p.complex.name} *data, ${p.scalar.name} coeff)
			{
				DEFINE_INDEXES;
				data[index] = complex_mul_scalar(data[index], coeff);
			}

			__kernel void multiply2(__global ${p.complex.name} *data1, __global ${p.complex.name} *data2,
				${p.scalar.name} c1, ${p.scalar.name} c2)
			{
				DEFINE_INDEXES;
				data1[index] = complex_mul_scalar(data1[index], c1);
				data2[index] = complex_mul_scalar(data2[index], c2);
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

			// Propagates state vector in k-space for steady state calculation (i.e., in imaginary time)
			// Version for processing two components at once
			__kernel void propagateKSpaceImaginaryTime2(
				__global ${p.complex.name} *data1, __global ${p.complex.name} *data2,
				read_only image3d_t kvectors)
			{
				DEFINE_INDEXES;

				${p.scalar.name} kvector = get_float_from_image(kvectors, i, j, k);

				${p.scalar.name} prop_coeff = native_exp(kvector *
					(${p.scalar.name})${-c.dt_steady / 2});

				data1[index] = complex_mul_scalar(data1[index], prop_coeff);
				data2[index] = complex_mul_scalar(data2[index], prop_coeff);
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

			// Propagates state in x-space for steady state calculation
			__kernel void propagateXSpaceTwoComponent(__global ${p.complex.name} *a,
				__global ${p.complex.name} *b, read_only image3d_t potentials)
			{
				DEFINE_INDEXES;

				${p.complex.name} a_res = a[index];
				${p.complex.name} b_res = b[index];

				//store initial x-space field
				${p.complex.name} a0 = a_res;
				${p.complex.name} b0 = b_res;

				${p.scalar.name} da, db, a_density, b_density;
				${p.scalar.name} V = get_float_from_image(potentials, i, j, k);

				//iterate to midpoint solution
				%for iter in range(c.itmax):
					//calculate midpoint log derivative and exponentiate
					a_density = squared_abs(a_res);
					b_density = squared_abs(b_res);

					da = exp((${p.scalar.name})${c.dt_steady / 2} *
						(-V - (${p.scalar.name})${c.g11} * a_density -
						(${p.scalar.name})${c.g12} * b_density));
					db = exp((${p.scalar.name})${c.dt_steady / 2} *
						(-V - (${p.scalar.name})${c.g12} * a_density -
						(${p.scalar.name})${c.g22} * b_density));

					//propagate to midpoint using log derivative
					a_res = complex_mul_scalar(a0, da);
					b_res = complex_mul_scalar(b0, db);
				%endfor

				//propagate to endpoint using log derivative
				a[index] = complex_mul_scalar(a_res, da);
				b[index] = complex_mul_scalar(b_res, db);
			}
		"""

		self._program = self._env.compileSource(kernel_template)

		self._kpropagate_func = FunctionWrapper(self._program.propagateKSpaceImaginaryTime)
		self._kpropagate2_func = FunctionWrapper(self._program.propagateKSpaceImaginaryTime2)
		self._xpropagate_func = FunctionWrapper(self._program.propagateXSpaceOneComponent)
		self._xpropagate2_func = FunctionWrapper(self._program.propagateXSpaceTwoComponent)
		self._multiply_func = FunctionWrapper(self._program.multiply)
		self._multiply2_func = FunctionWrapper(self._program.multiply2)

	def _cpu__kpropagate(self, two_component):
		if two_component:
			self._gs_a *= self._k_coeff
			self._gs_b *= self._k_coeff
		else:
			self._gs *= self._k_coeff

	def _gpu__kpropagate(self, two_component):
		if two_component:
			self._kpropagate2_func(self._env.queue, self._gs_a.shape, self._gs_a, self._gs_b, self._kvectors)
		else:
			self._kpropagate_func(self._env.queue, self._gs.shape, self._gs, self._kvectors)

	def _cpu__xpropagate(self, two_component):
		if two_component:
			a0 = self._gs_a.copy()
			b0 = self._gs_b.copy()

			for iter in xrange(self._env.constants.itmax):
				n_a = numpy.abs(self._gs_a) ** 2
				n_b = numpy.abs(self._gs_b) ** 2

				pa = self._potentials + n_a * self._env.constants.g11 + n_b * self._env.constants.g12
				pb = self._potentials + n_b * self._env.constants.g22 + n_a * self._env.constants.g12

				da = numpy.exp(pa * (-self._env.constants.dt_steady / 2))
				db = numpy.exp(pb * (-self._env.constants.dt_steady / 2))

				self._gs_a = a0 * da
				self._gs_b = b0 * db

			self._gs_a *= da
			self._gs_b *= db
		else:
			gs0 = self._gs.copy()
			for iter in xrange(self._env.constants.itmax):
				abs_gs = numpy.abs(self._gs)
				d_gs = numpy.exp((self._potentials + abs_gs * abs_gs * self._env.constants.g11) *
					(-self._env.constants.dt_steady / 2))
				self._gs = gs0 * d_gs
			self._gs *= d_gs

	def _gpu__xpropagate(self, two_component):
		if two_component:
			self._xpropagate2_func(self._env.queue, self._gs_a.shape, self._gs_a, self._gs_b, self._potentials)
		else:
			self._xpropagate_func(self._env.queue, self._gs.shape, self._gs, self._potentials)

	def _cpu__renormalize(self, coeff, two_component):
		if two_component:
			c1, c2 = coeff
			self._gs_a *= c1
			self._gs_b *= c2
		else:
			self._gs *= coeff

	def _gpu__renormalize(self, coeff, two_component):
		if two_component:
			c1, c2 = coeff
			self._multiply2_func(self._env.queue, self._gs_a.shape, self._gs_a, self._gs_b,
				self._env.precision.scalar.dtype(c1), self._env.precision.scalar.dtype(c2))
		else:
			self._multiply_func(self._env.queue, self._gs.shape, self._gs, self._env.precision.scalar.dtype(coeff))

	def _toXSpace(self, two_component):
		if two_component:
			self._plan.execute(self._gs_a)
			self._plan.execute(self._gs_b)
		else:
			self._plan.execute(self._gs)

	def _toKSpace(self, two_component):
		if two_component:
			self._plan.execute(self._gs_a, inverse=True)
			self._plan.execute(self._gs_b, inverse=True)
		else:
			self._plan.execute(self._gs, inverse=True)

	def create(self, two_component=False):

		if two_component:
			#self._gs_a = self._tf_gs.create(1)
			#self._gs_b = self._tf_gs.create(2)
			self._gs_a = self._env.toGPU(numpy.ones(self._env.constants.shape, self._env.precision.complex.dtype))
			self._gs_b = self._env.toGPU(numpy.ones(self._env.constants.shape, self._env.precision.complex.dtype))
			print self._gs_a.shape, self._gs_b.shape
		else:
			self._gs = self._tf_gs.create()

		stats = self._statistics

		E = 0

		if two_component:
			new_E = stats.countEnergyTwoComponent(self._gs_a, self._gs_b)
		else:
			new_E = stats.countEnergy(self._gs)

		self._toKSpace(two_component)

		while abs(E - new_E) / new_E > self._precision:

			# propagation

			self._kpropagate(two_component)
			self._toXSpace(two_component)
			self._xpropagate(two_component)
			self._toKSpace(two_component)
			self._kpropagate(two_component)

			# normalization

			self._toXSpace(two_component)

			# renormalize
			if two_component:
				N1 = stats.countParticles(self._gs_a, subtract_noise=False)
				N2 = stats.countParticles(self._gs_b, subtract_noise=False)
				c1 = math.sqrt(self._env.constants.N / (2 * N1))
				c2 = math.sqrt(self._env.constants.N / (2 * N2))
				self._renormalize((c1, c2), two_component)
			else:
				N = stats.countParticles(self._gs, subtract_noise=False)
				self._renormalize(math.sqrt(self._env.constants.N / N), two_component)

			E = new_E
			if two_component:
				new_E = stats.countEnergyTwoComponent(self._gs_a, self._gs_b)
				print new_E
			else:
				new_E = stats.countEnergy(self._gs)

			self._toKSpace(two_component)

		self._toXSpace(two_component)

		if two_component:
			print "Ground state calculation (two components):" + \
				" N = " + str(stats.countParticles(self._gs_a, subtract_noise=False)) + \
					" + " + str(stats.countParticles(self._gs_b, subtract_noise=False)) + \
				" E = " + str(stats.countEnergyTwoComponent(self._gs_a, self._gs_b)) + \
				" mu = " + str(stats.countMuTwoComponent(self._gs_a, self._gs_b))

			return self._gs_a, self._gs_b
		else:
			print "Ground state calculation (one component):" + \
				" N = " + str(stats.countParticles(self._gs, subtract_noise=False)) + \
				" E = " + str(stats.countEnergy(self._gs)) + \
				" mu = " + str(stats.countMu(self._gs))

			return self._gs
