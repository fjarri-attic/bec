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

	def __init__(self, gpu, precision, constants, mempool):
		PairedCalculation.__init__(self, gpu, mempool)
		self._precision = precision
		self._constants = constants
		self._mempool = mempool

		self._plan = createPlan(gpu, constants.nvx, constants.nvy, constants.nvz, precision)
		self._reduce = getReduce(gpu, precision, mempool)

		self._prepare()

	def _cpu__prepare(self):
		self._potentials = fillPotentialsArray(self._precision, self._constants)
		self._kvectors = fillKVectorsArray(self._precision, self._constants)

	def _cpu__getAverageDensity(self, state, subtract_noise=True):
		noise_term = self._constants.V / (2 * self._constants.dV) if subtract_noise else 0

		abs_values = numpy.abs(state)
		normalized_values = abs_values * abs_values - noise_term
		return self._reduce.sparse(normalized_values, self._constants.cells) / \
			(state.size / self._constants.cells) * self._constants.dV

	def _cpu_getStatesRatio(self, a, b):
		Na = self._reduce(self._getAverageDensity(a))
		Nb = self._reduce(self._getAverageDensity(b))
		return (Na - Nb) / (Na + Nb)

	def _cpu_countParticles(self, state, subtract_noise=True):
		return self._reduce(self._getAverageDensity(state, subtract_noise=subtract_noise))

	def _cpu__countState(self, state, coeff):
		kstate = self.allocate(state.shape, dtype=self._precision.complex.dtype)
		self._plan.execute(state, kstate, inverse=True, batch=state.size / self._constants.cells)

		res = self.allocate(state.shape, dtype=self._precision.scalar.dtype)

		n = numpy.abs(state) ** 2
		xk = state * kstate
		for e in xrange(state.size / self._constants.cells):
			start = e * self._constants.cells
			stop = (e + 1) * self._constants.cells
			res[start:stop,:,:] = numpy.abs(n[start:stop,:,:] * (self._potentials +
				n[start:stop,:,:] * (self._constants.g11 / coeff)) +
				xk[start:stop,:,:] * self._kvectors)

		return self._reduce(res) / (state.size / self._constants.cells) * self._constants.dV / self._constants.N

	def _cpu_countEnergy(self, state):
		return self._countState(state, 2)

	def _cpu_countMu(self, state):
		return self._countState(state, 1)

	def _gpu__prepare(self):
		kernel_template = """
			texture<${p.scalar.name}, 1> potentials;
			texture<${p.scalar.name}, 1> kvectors;

			__global__ void calculateDensity(${p.scalar.name} *res, ${p.complex.name} *state)
			{
				int index = GLOBAL_INDEX;
				res[index] = squared_abs(state[index]);
			}

			__global__ void calculateNoisedDensity(${p.scalar.name} *res, ${p.complex.name} *state)
			{
				int index = GLOBAL_INDEX;
				${p.scalar.name} noise_term = (${p.scalar.name})${0.5 * c.V / c.dV};
				res[index] = squared_abs(state[index]) - noise_term;
			}

			__global__ void calculateTwoStateNoisedDensity(${p.scalar.name} *a_res,
				${p.scalar.name} *b_res, ${p.complex.name} *a_state, ${p.complex.name} *b_state)
			{
				int index = GLOBAL_INDEX;
				${p.scalar.name} noise_term = (${p.scalar.name})${0.5 * c.V / c.dV};
				a_res[index] = squared_abs(a_state[index]) - noise_term;
				b_res[index] = squared_abs(b_state[index]) - noise_term;
			}

			%for name, coeff in (('Energy', 2), ('Mu', 1)):
				__global__ void calculate${name}(${p.scalar.name} *res,
					${p.complex.name} *xstate, ${p.complex.name} *kstate)
				{
					int index = GLOBAL_INDEX;
					${p.scalar.name} n_a = squared_abs(xstate[index]);
					${p.complex.name} differential = xstate[index] * kstate[index] * tex1Dfetch(kvectors, index);
					${p.scalar.name} nonlinear = n_a * (tex1Dfetch(potentials, index) +
						(${p.scalar.name})${c.g11} * n_a / ${coeff});

					// differential.y will be equal to 0, because \psi * D \psi is a real number
					res[index] = nonlinear + differential.x;
				}
			%endfor
		"""

		self._module = compileSource(kernel_template, self._precision, self._constants)

		self._calculate_mu = FunctionWrapper(self._module, "calculateMu", "PPP")
		self._calculate_energy = FunctionWrapper(self._module, "calculateEnergy", "PPP")
		self._calculate_density = FunctionWrapper(self._module, "calculateDensity", "PP")
		self._calculate_noised_density = FunctionWrapper(self._module, "calculateNoisedDensity", "PP")
		self._calculate_two_states_density = FunctionWrapper(self._module, "calculateTwoStateNoisedDensity", "PPPP")

		self._potentials_texref = self._module.get_texref("potentials")
		self._kvectors_texref = self._module.get_texref("kvectors")

		self._potentials_array = fillPotentialsTexture(self._precision, self._constants, self._potentials_texref)
		self._kvectors_array = fillKVectorsTexture(self._precision, self._constants, self._kvectors_texref)

	def _gpu_countParticles(self, state, subtract_noise=True):
		density = self.allocate(state.shape, self._precision.scalar.dtype)
		if subtract_noise:
			self._calculate_noised_density(state.size, density.gpudata, state.gpudata)
		else:
			self._calculate_density(state.size, density.gpudata, state.gpudata)
		return self._reduce(density) / (state.size / self._constants.cells) * self._constants.dV

	def _gpu_getStatesRatio(self, a, b):
		a_density = self.allocate(a.shape, self._precision.scalar.dtype)
		b_density = self.allocate(b.shape, self._precision.scalar.dtype)
		self._calculate_two_states_density(a.size, a_density.gpudata, b_density.gpudata,
			a.gpudata, b.gpudata)
		Na = self._reduce(a_density)
		Nb = self._reduce(b_density)
		return (Na - Nb) / (Na + Nb)

	def _gpu_countEnergy(self, state):
		kstate = self.allocate(state.shape, dtype=self._precision.complex.dtype)
		res = self.allocate(state.shape, dtype=self._precision.scalar.dtype)
		self._plan.execute(state, kstate, inverse=True, batch=state.size / self._constants.cells)
		self._calculate_energy(state.size, res.gpudata, state.gpudata, kstate.gpudata)
		return self._reduce(res) / (state.size / self._constants.cells) * self._constants.dV / self._constants.N

	def _gpu_countMu(self, state):
		kstate = self.allocate(state.shape, dtype=self._precision.complex.dtype)
		res = self.allocate(state.shape, dtype=self._precision.scalar.dtype)
		self._plan.execute(state, kstate, inverse=True, batch=state.size / self._constants.cells)
		self._calculate_mu(state.size, res.gpudata, state.gpudata, kstate.gpudata)
		return self._reduce(res) / (state.size / self._constants.cells) * self._constants.dV / self._constants.N


class VisibilityMeter(PairedCalculation):

	def __init__(self, gpu, precision, constants, mempool):

		PairedCalculation.__init__(self, gpu, mempool)
		self._precision = precision
		self._constants = constants

		self._statistics = ParticleStatistics(gpu, precision, constants, mempool)

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _gpu__prepare(self):

		kernels = """
			<%! import math %>

			// Pi/2 rotate around vector in equatorial plane, with angle alpha between it and x axis
			__global__ void halfPiRotate(${p.complex.name} *a_res, ${p.complex.name} *b_res, ${p.complex.name} *a,
				${p.complex.name} *b, ${p.scalar.name} alpha)
			{
				int index = GLOBAL_INDEX;

				${p.complex.name} a0 = a[index];
				${p.complex.name} b0 = b[index];

				${p.scalar.name} cosa = cos(alpha);
				${p.scalar.name} sina = sin(alpha);

				a_res[index] = (a0 + b0 * ${p.complex.ctr}(sina, -cosa)) *
					(${p.scalar.name})${math.sqrt(0.5)};
				b_res[index] = (a0 * ${p.complex.ctr}(-sina, -cosa) + b0) *
					(${p.scalar.name})${math.sqrt(0.5)};
			}
		"""

		self._module = compileSource(kernels, self._precision, self._constants)
		self._half_pi_rotate_func = FunctionWrapper(self._module, "halfPiRotate",
			"PPPP" + self._precision.scalar.ctype)

	def _gpu__halfPiRotate(self, a_buffer, b_buffer, a, b, alpha):
		self._half_pi_rotate_func(a.size, a_buffer.gpudata, b_buffer.gpudata, a.gpudata, b.gpudata, alpha)

	def _cpu__halfPiRotate(self, a_buffer, b_buffer, a, b, alpha):
		coeff1 = math.sin(alpha) - 1j * math.cos(alpha)
		coeff2 = -math.sin(alpha) - 1j * math.cos(alpha)
		a_buffer[:,:,:] = (a + b * coeff1) * math.sqrt(0.5)
		b_buffer[:,:,:] = (a * coeff2 + b) * math.sqrt(0.5)

	def getPoints(self, a, b, num_points):
		res = numpy.empty(num_points, dtype=self._precision.scalar.dtype)

		a_buffer = self.allocate(a.shape, a.dtype)
		b_buffer = self.allocate(b.shape, b.dtype)

		for i in xrange(num_points):
			alpha = 2 * math.pi * i / num_points
			self._halfPiRotate(a_buffer, b_buffer, a, b, alpha)
			res[i] = abs(self._statistics.getStatesRatio(a_buffer, b_buffer))
		return res

	def get(self, a, b):
		return numpy.max(self.getPoints(a, b, 5))
