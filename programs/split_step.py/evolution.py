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
from ground_state import *


class TwoComponentBEC(PairedCalculation):

	def __init__(self, gpu, precision, constants, mempool):
		PairedCalculation.__init__(self, gpu)
		self._precision = precision
		self._constants = constants
		self._mempool = mempool

		self._gs = GPEGroundState(gpu, precision, constants, mempool)
		self._plan = createPlan(gpu, constants.nvx, constants.nvy, constants.nvz, precision)
		self._stats = ParticleStatistics(gpu, precision, constants, mempool)

		self._prepare()
		self.reset()

	def _gpu__prepare(self):

		kernels = Template("""
			<%!
				from math import sqrt
			%>

			texture<${p.scalar.name}, 1> potentials;
			texture<${p.scalar.name}, 1> kvectors;

			// Initialize ensembles with steady state + noise for evolution calculation
			__global__ void initializeEnsembles(${p.complex.name} *a, ${p.complex.name} *b,
				${p.complex.name} *steady_state, ${p.complex.name} *noise)
			{
				int index = GLOBAL_INDEX;

				${p.complex.name} noise_a = noise[index];
				${p.complex.name} noise_b = noise[index + ${c.cells * c.ensembles}];
				${p.complex.name} steady_val = steady_state[index % ${c.cells}];

				${p.scalar.name} coeff = (${p.scalar.name})${1.0 / sqrt(c.dV)};

				//Initialises a-ensemble amplitudes with vacuum noise
				a[index] = steady_val + noise_a * (${c.V1} * coeff);

				//Initialises b-ensemble amplitudes with vacuum noise
				b[index] = noise_b * (${c.V2} * coeff);
			}

			// Propagates state vector in k-space for evolution calculation (i.e., in real time)
			__global__ void propagateKSpaceRealTime(${p.complex.name} *a, ${p.complex.name} *b, ${p.scalar.name} dt)
			{
				int index = GLOBAL_INDEX;
				<% total_pow = c.nvx_pow + c.nvy_pow + c.nvz_pow %>
				int index_in_ensemble = index - ((index >> ${total_pow}) << ${total_pow});

				${p.scalar.name} prop_angle = tex1Dfetch(kvectors, index_in_ensemble) * dt / 2;
				${p.complex.name} prop_coeff = ${p.complex.ctr}(cos(prop_angle), sin(prop_angle));

				${p.complex.name} a0 = a[index];
				${p.complex.name} b0 = b[index];

				a[index] = a0 * prop_coeff;
				b[index] = b0 * prop_coeff;
			}

			// Propagates state vector in x-space for evolution calculation
			__global__ void propagateXSpaceTwoComponent(${p.complex.name} *aa, ${p.complex.name} *bb, ${p.scalar.name} dt)
			{
				int index = GLOBAL_INDEX;

				<% total_pow = c.nvx_pow + c.nvy_pow + c.nvz_pow %>
				${p.scalar.name} V = tex1Dfetch(potentials, index - ((index >> ${total_pow}) << ${total_pow}));

				${p.complex.name} a = aa[index];
				${p.complex.name} b = bb[index];

				//store initial x-space field
				${p.complex.name} a0 = a;
				${p.complex.name} b0 = b;

				${p.complex.name} pa, pb, da = ${p.complex.ctr}(0, 0), db = ${p.complex.ctr}(0, 0);
				${p.scalar.name} n_a, n_b;

				//iterate to midpoint solution
				%for iter in range(c.itmax):
					n_a = squared_abs(a);
					n_b = squared_abs(b);

					// TODO: there must be no minus sign before imaginary part,
					// but without it the whole thing diverges
					pa = ${p.complex.ctr}(
						-(${c.l111} * n_a * n_a + ${c.l12} * n_b) / 2,
						-(-V - ${c.g11} * n_a - ${c.g12} * n_b));
					pb = ${p.complex.ctr}(
						-(${c.l22} * n_b + ${c.l12} * n_a) / 2,
						-(-V - ${c.g22} * n_b - ${c.g12} * n_a + ${c.detuning}));

					// calculate midpoint log derivative and exponentiate
					da = cexp(pa * (dt / 2));
					db = cexp(pb * (dt / 2));

					//propagate to midpoint using log derivative
					a = a0 * da;
					b = b0 * db;
				%endfor

				//propagate to endpoint using log derivative
				aa[index] = a * da;
				bb[index] = b * db;
			}

			// Apply pi/2 pulse (instantaneus approximation)
			__global__ void applyHalfPiPulse(${p.complex.name} *a, ${p.complex.name} *b)
			{
				int index = GLOBAL_INDEX;

				${p.complex.name} a0 = a[index];
				${p.complex.name} b0 = b[index];

				a[index] = (a0 + b0 * ${p.complex.ctr}(0, -1)) * sqrt(0.5);
				b[index] = (a0 * ${p.complex.ctr}(0, -1) + b0) * sqrt(0.5);
			}

			// Apply pi pulse (instantaneus approximation)
			__global__ void applyPiPulse(${p.complex.name} *a, ${p.complex.name} *b)
			{
				int index = GLOBAL_INDEX;

				${p.complex.name} a0 = a[index];
				${p.complex.name} b0 = b[index];

				a[index] = b0 * ${p.complex.ctr}(0, -1);
				b[index] = a0 * ${p.complex.ctr}(0, -1);
			}

			// Pi/2 rotate around vector in equatorial plane, with angle alpha between it and x axis
			__global__ void halfPiRotate(${p.scalar.name} *a_res, ${p.scalar.name} *b_res, ${p.complex.name} *a,
				${p.complex.name} *b, ${p.scalar.name} alpha)
			{
				int index = GLOBAL_INDEX;

				${p.complex.name} a0 = a[index];
				${p.complex.name} b0 = b[index];

				${p.scalar.name} cosa = cos(alpha);
				${p.scalar.name} sina = sin(alpha);

				${p.complex.name} a_new = (a0 + b0 * ${p.complex.ctr}(sina, -cosa)) * ${sqrt(0.5)};
				${p.complex.name} b_new = (a0 * ${p.complex.ctr}(-sina, -cosa) + b0) * ${sqrt(0.5)};

				a_res[index] = squared_abs(a_new) - ${c.V / (2 * c.dV)};
				b_res[index] = squared_abs(b_new) - ${c.V / (2 * c.dV)};
			}
		""")

		self._module = compileSource(kernels, self._precision, self._constants)
		self._init_ensembles = FunctionWrapper(self._module, "initializeEnsembles", "PPPP")
		self._kpropagate = FunctionWrapper(self._module, "propagateKSpaceRealTime", "PP" + self._precision.scalar.ctype)
		self._xpropagate = FunctionWrapper(self._module, "propagateXSpaceTwoComponent", "PP" + self._precision.scalar.ctype)
		self._half_pi_pulse = FunctionWrapper(self._module, "applyHalfPiPulse", "PP")

		self._potentials_texref = self._module.get_texref("potentials")
		self._kvectors_texref = self._module.get_texref("kvectors")

		self._potentials_array = fillPotentialsTexture(self._precision, self._constants, self._potentials_texref)
		self._kvectors_array = fillKVectorsTexture(self._precision, self._constants, self._kvectors_texref)

	def _to_kspace(self):
		self._plan.execute(self._a, batch=self._constants.ensembles, inverse=True)
		self._plan.execute(self._b, batch=self._constants.ensembles, inverse=True)

	def _to_xspace(self):
		self._plan.execute(self._a, batch=self._constants.ensembles)
		self._plan.execute(self._b, batch=self._constants.ensembles)

	def _gpu_reset(self):
		size = self._constants.cells * self._constants.ensembles
		randoms = (numpy.random.normal(scale=0.5, size=2 * size) +
			1j * numpy.random.normal(scale=0.5, size=2 * size)).astype(self._precision.complex.dtype)
		randoms_gpu = gpuarray.to_gpu(randoms)
		gs = self._gs.create()

		self._a = gpuarray.GPUArray(self._constants.ens_shape, self._precision.complex.dtype, self._mempool)
		self._b = gpuarray.GPUArray(self._constants.ens_shape, self._precision.complex.dtype, self._mempool)
		self._init_ensembles(size, self._a.gpudata, self._b.gpudata, gs.gpudata, randoms_gpu.gpudata)

		self._to_kspace()

		# equilibration
		if self._constants.t_equilib > 0:
			for t in xrange(0, self._constants.t_equilib, self._constants.dt_evo):
				self.propagate(self._constants.dt_evo)

		self._t = 0

		# first pi/2 pulse
		self._half_pi_pulse(size, self._a.gpudata, self._b.gpudata)

	def _gpu_propagate(self, dt):

		size = self._constants.cells * self._constants.ensembles

		self._kpropagate(size, self._a.gpudata, self._b.gpudata, dt)
		self._to_xspace()
		self._xpropagate(size, self._a.gpudata, self._b.gpudata, dt)
		self._to_kspace()
		self._kpropagate(size, self._a.gpudata, self._b.gpudata, dt)

		self._t += dt

	def _cpu__prepare(self):
		potentials = fillPotentialsArray(self._precision, self._constants)
		kvectors = fillKVectorsArray(self._precision, self._constants)

		self._potentials = numpy.empty(self._constants.ens_shape, dtype=self._precision.complex.dtype)
		self._kvectors = numpy.empty(self._constants.ens_shape, dtype=self._precision.complex.dtype)

		for e in range(self._constants.ensembles):
			start = e * self._constants.nvz
			stop = (e + 1) * self._constants.nvz

			self._potentials[start:stop,:,:] = potentials
			self._kvectors[start:stop,:,:] = kvectors

	def _cpu_reset(self):
		gs = self._gs.create()

		coeff = 1.0 / math.sqrt(self._constants.dV)
		self._a = numpy.empty(self._constants.ens_shape, dtype=self._precision.complex.dtype)
		self._b = numpy.empty(self._constants.ens_shape, dtype=self._precision.complex.dtype)

		for e in range(self._constants.ensembles):
			start = e * self._constants.nvz
			stop = (e + 1) * self._constants.nvz

			self._a[start:stop,:,:] = gs + numpy.random.normal(scale=0.5, size=self._constants.shape) * coeff * self._constants.V1 + \
				1j * numpy.random.normal(scale=0.5, size=self._constants.shape) * coeff * self._constants.V1
			self._b[start:stop,:,:] = numpy.random.normal(scale=0.5, size=self._constants.shape) * coeff * self._constants.V2 + \
				1j * numpy.random.normal(scale=0.5, size=self._constants.shape) * coeff * self._constants.V2

		self._to_kspace()

		# equilibration
		if self._constants.t_equilib > 0:
			for t in xrange(0, self._constants.t_equilib, self._constants.dt_evo):
				self.propagate(self._constants.dt_evo)

		self._t = 0

		# TODO: where should pi/2 pulse occur - in x-space or in k-space?

		# first pi/2 pulse
		a0 = self._a.copy()
		b0 = self._b.copy()
		self._a = (a0 - 1j * b0) * math.sqrt(0.5)
		self._b = (b0 - 1j * a0) * math.sqrt(0.5)

	def _cpu_propagate(self, dt):
		size = self._constants.cells * self._constants.ensembles
		kcoeff = numpy.exp(1j * self._kvectors * (-dt / 2))

		self._a *= kcoeff
		self._b *= kcoeff

		self._to_xspace()

		a0 = self._a.copy()
		b0 = self._b.copy()

		for iter in xrange(self._constants.itmax):
			n_a = numpy.abs(self._a)
			n_b = numpy.abs(self._b)

			n_a = n_a * n_a
			n_b = n_b * n_b

			pa = -(n_a * n_a * self._constants.l111 + n_b * self._constants.l12) / 2 + \
				1j * (-(-self._potentials - n_a * self._constants.g11 - n_b * self._constants.g12))
			pb = -(n_b * self._constants.l22 + n_a * self._constants.l12) / 2 + \
				1j * (-(-self._potentials - n_b * self._constants.g22 - n_a * self._constants.g12 + self._constants.detuning))

			da = numpy.exp(pa * (dt / 2))
			db = numpy.exp(pb * (dt / 2))

			self._a = a0 * da
			self._b = b0 * db

		self._a *= da
		self._b *= db

		self._to_kspace()

		self._a *= kcoeff
		self._b *= kcoeff

		self._t += dt

	def _runCallbacks(self, callbacks):
		# transform to x-space
		self._plan.execute(self._a, batch=self._constants.ensembles)
		self._plan.execute(self._b, batch=self._constants.ensembles)

		for callback in callbacks:
			callback(self._t * self._constants.t_rho, self._a, self._b)

		# transform to k-space
		self._plan.execute(self._a, batch=self._constants.ensembles, inverse=True)
		self._plan.execute(self._b, batch=self._constants.ensembles, inverse=True)


	def runEvolution(self, tstop, callbacks, callback_dt=0):
		self._t = 0
		callback_t = 0
		while self._t * self._constants.t_rho < tstop:
			self.propagate(self._constants.dt_evo)
			callback_t += self._constants.dt_evo * self._constants.t_rho

			if callback_t > callback_dt:
				self._runCallbacks(callbacks)
				callback_t = 0

		self._runCallbacks(callbacks)
