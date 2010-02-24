import math
import copy
from mako.template import Template
import numpy

try:
	import pycuda.driver as cuda
	from pycuda.compiler import SourceModule
	from pycuda import gpuarray
except:
	pass

from globals import *
from fft import createPlan
from reduce import getReduce


class PairedCalculation:
	def __init__(self, gpu):
		prefix = "_gpu_" if gpu else "_cpu_"

		for attr in dir(self):
			if attr.startswith(prefix):
				name = attr[len(prefix):]
				self.__dict__[name] = getattr(self, attr)


class TFGroundState(PairedCalculation):

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
		kernel_template = Template("""
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
		""")

		self._module = compileSource(kernel_template, self._precision, self._constants)
		self._func = FunctionWrapper(self._module, "fillWithTFGroundState", "P")
		self._texref = self._module.get_texref("potentials")
		fillPotentialsTexture(self._precision, self._constants, self._texref)

	def _gpu_create(self):
		res = gpuarray.GPUArray(self._constants.shape, self._precision.complex.dtype, allocator=self._mempool)
		self._func(self._constants.cells, res.gpudata)
		return res


class ParticleStatistics(PairedCalculation):

	def __init__(self, gpu, precision, constants, mempool):
		PairedCalculation.__init__(self, gpu)
		self._precision = precision
		self._constants = copy.deepcopy(constants)
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
		kernel_template = Template("""
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
		""")

		self._module = compileSource(kernel_template, self._precision, self._constants)

		self._calculate_mu = FunctionWrapper(self._module, "calculateMu", "PPP")
		self._calculate_energy = FunctionWrapper(self._module, "calculateEnergy", "PPP")
		self._calculate_density = FunctionWrapper(self._module, "calculateDensity", "PP")
		self._calculate_noised_density = FunctionWrapper(self._module, "calculateNoisedDensity", "PP")

		self._potentials_texref = self._module.get_texref("potentials")
		self._kvectors_texref = self._module.get_texref("kvectors")

		fillPotentialsTexture(self._precision, self._constants, self._potentials_texref)
		fillKVectorsTexture(self._precision, self._constants, self._kvectors_texref)

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


class GPEGroundState(PairedCalculation):

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

	def _cpu__prepare(self):
		self._potentials = fillPotentialsArray(self._precision, self._constants)
		self._kvectors = fillKVectorsArray(self._precision, self._constants)

	def _cpu_create(self):
		gs = self._tf_gs.create()

		E = 0
		new_E = self._statistics.countEnergy(gs)

		self._plan.execute(gs, inverse=True)

		while abs(E - new_E) / E > 1e-6:
			E = new_E

			gs *= numpy.exp(self._kvectors * (-self._constants.dt_steady / 2)) # k-space propagation
			self._plan.execute(gs) # FFT to x-space

			# x-space propagation
			gs0 = gs.copy()
			for iter in xrange(self._constants.itmax):
				abs_gs = numpy.abs(gs)
				d_gs = numpy.exp((self._potentials + abs_gs * abs_gs * self._constants.g11) *
					(-self._constants.dt_steady / 2))
				gs = gs0 * d_gs
			gs *= d_gs

			self._plan.execute(gs, inverse=True) # FFT to k-space
			gs *= numpy.exp(self._kvectors * (-self._constants.dt_steady / 2)) # k-space propagation
			self._plan.execute(gs) # FFT to x-space

			# renormalize
			N = self._statistics.countParticles(gs, subtract_noise=False)
			gs *= math.sqrt(self._constants.N / N)

			new_E = self._statistics.countEnergy(gs)

			self._plan.execute(gs, inverse=True)

		self._plan.execute(gs) # FFT to x-state

		print "N = " + str(self._statistics.countParticles(gs, subtract_noise=False))
		print "E = " + str(self._statistics.countEnergy(gs))
		print "mu = " + str(self._statistics.countMu(gs))

	def _gpu_create(self):
		tf_gs = self._tf_gs.create()
		print "N = " + str(self._statistics.countParticles(tf_gs, subtract_noise=False))
		print "E = " + str(self._statistics.countEnergy(tf_gs))
		print "mu = " + str(self._statistics.countMu(tf_gs))

	def _gpu__prepare(self):
		kernel_template = Template("""
			texture<${p.scalar.name}, 1> potentials;
			texture<${p.scalar.name}, 1> kvectors;

			// Propagates state vector in k-space for steady state calculation (i.e., in imaginary time)
			__global__ void propagateKSpaceImaginaryTime(${p.complex.name} *data)
			{
				int index = GLOBAL_INDEX;
				${p.scalar.name} prop_coeff = exp(-${c.dt_steady} / 2 * tex1Dfetch(kvectors, index));
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
				for(int iter = 0; iter < ${c.itmax}; iter++)
				{
					//calculate midpoint log derivative and exponentiate
					da = exp(${c.dt_steady} / 2 * (-V - ${c.g11} * squared_abs(a)));

					//propagate to midpoint using log derivative
					a = a0 * da;
				}

				//propagate to endpoint using log derivative
				data[index] = a * da;
			}
		""")

		defines = KERNEL_DEFINES.render(p=self._precision)
		kernel_src = kernel_template.render(p=self._precision, c=self._constants)

		self._module = SourceModule(defines + kernel_src, no_extern_c=True)

		#self._kpropagate = FunctionWrapper(self._module.get_function("propagateKSpaceImaginaryTime"),
		#	"P", self._constants.cells)
		#self._xpropagate = FunctionWrapper(self._module.get_function("propagateXSpaceOneComponent"),
		#	"P", self._constants.cells)
		self._potentials_texref = self._module.get_texref("potentials")
		self._kvectors_texref = self._module.get_texref("kvectors")
