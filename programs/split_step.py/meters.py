"""
Different meters for particle states (measuring particles number, energy and so on)
"""

from globals import *
from fft import createPlan
from reduce import getReduce


class ParticleStatistics(PairedCalculation):
	"""
	Calculates number of particles, energy per particle or
	chemical potential per particle for given state.
	"""

	def __init__(self, gpu, precision, constants, mempool):
		PairedCalculation.__init__(self, gpu)
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
		res = numpy.zeros(self._constants.shape, dtype=self._precision.scalar.dtype)
		noise_term = self._constants.V / (2 * self._constants.dV) if subtract_noise else 0

		abs_values = numpy.abs(state)
		normalized_values = abs_values * abs_values - noise_term
		return self._reduce.sparse(normalized_values, self._constants.cells) / \
			(state.size / self._constants.cells) * self._constants.dV

	def _cpu_countParticles(self, state, subtract_noise=True):
		return self._reduce(self._getAverageDensity(state, subtract_noise=subtract_noise))

	def _cpu__countState(self, state, coeff):
		kstate = numpy.empty(state.shape, dtype=self._precision.complex.dtype)
		self._plan.execute(state, kstate, inverse=True, batch=state.size / self._constants.cells)

		res = numpy.empty(state.shape, dtype=self._precision.scalar.dtype)

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
				${p.scalar.name} noise_term = 0.5 * ${c.V} / (${c.dx} * ${c.dy} * ${c.dz});
				res[index] = squared_abs(state[index]) - noise_term;
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

		self._potentials_texref = self._module.get_texref("potentials")
		self._kvectors_texref = self._module.get_texref("kvectors")

		self._potentials_array = fillPotentialsTexture(self._precision, self._constants, self._potentials_texref)
		self._kvectors_array = fillKVectorsTexture(self._precision, self._constants, self._kvectors_texref)

	def _gpu_countParticles(self, state, subtract_noise=True):
		density = gpuarray.GPUArray(state.shape, self._precision.scalar.dtype, allocator=self._mempool)
		if subtract_noise:
			self._calculate_noised_density(state.size, density.gpudata, state.gpudata)
		else:
			self._calculate_density(state.size, density.gpudata, state.gpudata)
		return self._reduce(density) / (state.size / self._constants.cells) * self._constants.dV

	def _gpu_countEnergy(self, state):
		kstate = gpuarray.GPUArray(state.shape, dtype=self._precision.complex.dtype, allocator=self._mempool)
		res = gpuarray.GPUArray(state.shape, dtype=self._precision.scalar.dtype, allocator=self._mempool)
		self._plan.execute(state, kstate, inverse=True, batch=state.size / self._constants.cells)
		self._calculate_energy(state.size, res.gpudata, state.gpudata, kstate.gpudata)
		return self._reduce(res) / (state.size / self._constants.cells) * self._constants.dV / self._constants.N

	def _gpu_countMu(self, state):
		kstate = gpuarray.GPUArray(state.shape, dtype=self._precision.complex.dtype, allocator=self._mempool)
		res = gpuarray.GPUArray(state.shape, dtype=self._precision.scalar.dtype, allocator=self._mempool)
		self._plan.execute(state, kstate, inverse=True, batch=state.size / self._constants.cells)
		self._calculate_mu(state.size, res.gpudata, state.gpudata, kstate.gpudata)
		return self._reduce(res) / (state.size / self._constants.cells) * self._constants.dV / self._constants.N
