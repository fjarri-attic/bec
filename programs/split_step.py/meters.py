"""
Different meters for particle states (measuring particles number, energy and so on)
"""

import math

from globals import *
from fft import createPlan
from reduce import getReduce


class ParticleStatistics(PairedCalculation):
	"""
	Calculates number of particles, energy per particle or
	chemical potential per particle for given state.
	"""

	def __init__(self, env):
		PairedCalculation.__init__(self, env)
		self._env = env

		self._plan = createPlan(env, env.constants.nvx, env.constants.nvy, env.constants.nvz)
		self._reduce = getReduce(env)

		self._potentials = getPotentials(self._env)
		self._kvectors = getKVectors(self._env)

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _cpu_getAverageDensity(self, state, subtract_noise=True):
		if subtract_noise:
			noise_term = self._env.constants.V / (2 * self._env.constants.dV)
		else:
			noise_term = 0

		abs_values = numpy.abs(state)
		normalized_values = abs_values * abs_values - noise_term
		return self._reduce.sparse(normalized_values, self._env.constants.cells) / \
			(state.size / self._env.constants.cells)

	def _cpu__countState(self, state, coeff):
		kstate = self._env.allocate(state.shape, dtype=self._env.precision.complex.dtype)
		self._plan.execute(state, kstate, inverse=True, batch=state.size / self._env.constants.cells)

		res = self._env.allocate(state.shape, dtype=self._env.precision.scalar.dtype)

		n = numpy.abs(state) ** 2
		xk = state * kstate
		for e in xrange(state.size / self._env.constants.cells):
			start = e * self._env.constants.cells
			stop = (e + 1) * self._env.constants.cells
			res[start:stop,:,:] = numpy.abs(n[start:stop,:,:] * (self._potentials +
				n[start:stop,:,:] * (self._env.constants.g11 / coeff)) +
				xk[start:stop,:,:] * self._kvectors)

		return self._reduce(res) / (state.size / self._env.constants.cells) * self._env.constants.dV / self._env.constants.N

	def _cpu_countEnergy(self, state):
		return self._countState(state, 2)

	def _cpu_countMu(self, state):
		return self._countState(state, 1)

	def _cpu_getVisibility(self, a, b):
		Na = self._reduce(self.getAverageDensity(a)) * self._env.constants.dV
		Nb = self._reduce(self.getAverageDensity(b)) * self._env.constants.dV

		coeff = self._env.constants.dV / (a.size / self._env.constants.cells)

		interaction = abs(self._reduce(a * numpy.conj(b))) * coeff

		return 2 * interaction / (Na + Nb)

	def _gpu__prepare(self):
		kernel_template = """
			__kernel void calculateDensity(__global ${p.scalar.name} *res,
				__global ${p.complex.name} *state, int ensembles)
			{
				DEFINE_INDEXES;
				res[index] = squared_abs(state[index]) / ensembles;
			}

			__kernel void calculateNoisedDensity(__global ${p.scalar.name} *res,
				__global ${p.complex.name} *state, int ensembles)
			{
				DEFINE_INDEXES;
				${p.scalar.name} noise_term = (${p.scalar.name})${0.5 * c.V / c.dV};
				res[index] = (squared_abs(state[index]) - noise_term) / ensembles;
			}

			__kernel void calculateTwoStatesDensity(__global ${p.scalar.name} *a_res,
				__global ${p.scalar.name} *b_res, __global ${p.complex.name} *a_state,
				__global ${p.complex.name} *b_state)
			{
				DEFINE_INDEXES;
				${p.scalar.name} noise_term = (${p.scalar.name})${0.5 * c.V / c.dV};
				a_res[index] = squared_abs(a_state[index]) - noise_term;
				b_res[index] = squared_abs(b_state[index]) - noise_term;
			}

			__kernel void calculateTwoStatesInteraction(__global ${p.complex.name} *interaction,
				__global ${p.complex.name} *a_state, __global ${p.complex.name} *b_state)
			{
				DEFINE_INDEXES;
				interaction[index] = complex_mul(a_state[index], conj(b_state[index]));
			}

			%for name, coeff in (('Energy', 2), ('Mu', 1)):
				__kernel void calculate${name}(__global ${p.scalar.name} *res,
					__global ${p.complex.name} *xstate, __global ${p.complex.name} *kstate,
					read_only image3d_t potentials, read_only image3d_t kvectors)
				{
					DEFINE_INDEXES;

					${p.scalar.name} potential = get_float_from_image(potentials, i, j, k);
					${p.scalar.name} kvector = get_float_from_image(kvectors, i, j, k);

					${p.scalar.name} n_a = squared_abs(xstate[index]);
					${p.complex.name} differential =
						complex_mul(complex_mul(xstate[index], kstate[index]), kvector);
					${p.scalar.name} nonlinear = n_a * (potential +
						(${p.scalar.name})${c.g11} * n_a / ${coeff});

					// differential.y will be equal to 0, because \psi * D \psi is a real number
					res[index] = nonlinear + differential.x;
				}
			%endfor
		"""

		self._program = self._env.compileSource(kernel_template)

		self._calculate_mu = FunctionWrapper(self._program.calculateMu)
		self._calculate_energy = FunctionWrapper(self._program.calculateEnergy)
		self._calculate_density = FunctionWrapper(self._program.calculateDensity)
		self._calculate_noised_density = FunctionWrapper(self._program.calculateNoisedDensity)
		self._calculate_two_states_density = FunctionWrapper(self._program.calculateTwoStatesDensity)
		self._calculate_two_states_interaction = FunctionWrapper(self._program.calculateTwoStatesInteraction)

	def _gpu_getAverageDensity(self, state, subtract_noise=True):
		density = self._env.allocate(state.shape, self._env.precision.scalar.dtype)
		ensembles = state.size / self._env.constants.cells
		if subtract_noise:
			self._calculate_noised_density(self._env.queue, state.shape, density, state, numpy.int32(ensembles))
		else:
			self._calculate_density(self._env.queue, state.shape, density, state, numpy.int32(ensembles))
		return self._reduce.sparse(density, final_length=self._env.constants.cells)

	def _gpu_countEnergy(self, state):
		kstate = self._env.allocate(state.shape, dtype=self._env.precision.complex.dtype)
		res = self._env.allocate(state.shape, dtype=self._env.precision.scalar.dtype)
		self._plan.execute(state, kstate, inverse=True, batch=state.size / self._env.constants.cells)
		self._calculate_energy(self._env.queue, state.shape, res, state, kstate, self._potentials, self._kvectors)
		return self._reduce(res) / (state.size / self._env.constants.cells) * self._env.constants.dV / self._env.constants.N

	def _gpu_countMu(self, state):
		kstate = self._env.allocate(state.shape, dtype=self._env.precision.complex.dtype)
		res = self._env.allocate(state.shape, dtype=self._env.precision.scalar.dtype)
		self._plan.execute(state, kstate, inverse=True, batch=state.size / self._env.constants.cells)
		self._calculate_mu(self._env.queue, state.shape, res, state, kstate, self._potentials, self._kvectors)
		return self._reduce(res) / (state.size / self._env.constants.cells) * self._env.constants.dV / self._env.constants.N

	def _gpu_getVisibility(self, a, b):
		a_density = self._env.allocate(a.shape, self._env.precision.scalar.dtype)
		b_density = self._env.allocate(b.shape, self._env.precision.scalar.dtype)

		interaction = self._env.allocate(a.shape, self._env.precision.complex.dtype)

		self._calculate_two_states_density(self._env.queue, a.shape, a_density, b_density, a, b)
		self._calculate_two_states_interaction(self._env.queue, a.shape, interaction, a, b)

		Na = self._reduce(a_density)
		Nb = self._reduce(b_density)

		interaction = self._reduce(interaction)

		return 2 * abs(interaction) / (Na + Nb)

	def countParticles(self, state, subtract_noise=True):
		return self._reduce(self.getAverageDensity(state, subtract_noise=subtract_noise)) * self._env.constants.dV


class BlochSphereProjection(PairedCalculation):

	def __init__(self, env):
		PairedCalculation.__init__(self, env)
		self._env = env

	def _cpu_getProjection(self, a, b, amp_points, phase_points, amp_range, phase_range):

		amp_min, amp_max = amp_range
		phase_min, phase_max = phase_range

		res = numpy.zeros((amp_points, phase_points), dtype=self._env.precision.scalar.dtype)

		a = a.ravel()
		b = b.ravel()

		amp_a = numpy.abs(a)
		amp_b = numpy.abs(b)
		phase_a = numpy.angle(a)
		phase_b = numpy.angle(b)

		max_a = numpy.max(amp_a)
		max_b = numpy.max(amp_b)

		d_amp = (amp_max - amp_min) / (amp_points - 1)
		d_phase = (phase_max - phase_min) / (phase_points - 1)

		phase_diff = phase_b - phase_a
		for i in xrange(a.size):
			if phase_diff[i] < 0:
				phase_diff[i] += 2 * math.pi

		amp_diff = ((2 * numpy.arctan(amp_b / amp_a) - amp_min) / d_amp).astype(numpy.int32)
		phase_diff = ((phase_diff - phase_min) / d_phase).astype(numpy.int32)

		for i in xrange(a.size):
			# removing noise
			if amp_a[i] < max_a * 1e-6 and amp_b[i] < max_b * 1e-6:
				continue

			amp_coord = amp_diff[i]
			phase_coord = phase_diff[i]

			if amp_coord < 0 or amp_coord >= amp_points or phase_coord < 0 or phase_coord >= phase_points:
				continue

			res[amp_coord, phase_coord] += 1

		return res


class Projection:

	def __init__(self, env):
		self._env = env
		self._reduce = getReduce(env)
		self._stats = ParticleStatistics(env)

	def getXY(self, state):
		density = self._stats.getAverageDensity(state)
		x = self._env.constants.nvx
		y = self._env.constants.nvy
		return self._env.toCPU(self._reduce.sparse(density, final_length=x * y), shape=(y, x))

	def getYZ(self, state):
		density = self._stats.getAverageDensity(state)
		y = self._env.constants.nvy
		z = self._env.constants.nvz
		return self._env.toCPU(self._reduce(density, final_length=y * z), shape=(z, y))

	def getZ(self, state):
		density = self._stats.getAverageDensity(state)
		z = self._env.constants.nvz
		return self._env.toCPU(self._reduce(density, final_length=z))
