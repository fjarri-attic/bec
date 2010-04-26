"""
Classes, modeling the evolution of BEC.
"""

import math
import copy
from mako.template import Template
import numpy

try:
	import pyopencl as cl
except:
	pass

from globals import *
from fft import createPlan
from reduce import getReduce
from ground_state import GPEGroundState


class Pulse(PairedCalculation):

	def __init__(self, env):
		self._env = env
		PairedCalculation.__init__(self, env)
		self._prepare()

	def _cpu__prepare(self):
		pass

	def _gpu__prepare(self):
		kernels = """
			// Apply pi/2 pulse (instantaneus approximation)
			__kernel void applyHalfPiPulse(__global ${p.complex.name} *a, __global ${p.complex.name} *b)
			{
				DEFINE_INDEXES;

				${p.complex.name} a0 = a[index];
				${p.complex.name} b0 = b[index];

				${p.complex.name} minus_i = ${p.complex.ctr}(0, -1);
				a[index] = complex_mul_scalar(a0 + complex_mul(b0, minus_i), (${p.scalar.name})${sqrt(0.5)});
				b[index] = complex_mul_scalar(complex_mul(a0, minus_i) + b0, (${p.scalar.name})${sqrt(0.5)});
			}

			// Apply pi pulse (instantaneus approximation)
			__kernel void applyPiPulse(__global ${p.complex.name} *a, __global ${p.complex.name} *b)
			{
				DEFINE_INDEXES;

				${p.complex.name} a0 = a[index];
				${p.complex.name} b0 = b[index];

				${p.complex.name} minus_i = ${p.complex.ctr}(0, -1);

				a[index] = complex_mul(b0, minus_i);
				b[index] = complex_mul(a0, minus_i);
			}
		"""

		self._program = self._env.compileSource(kernels, sqrt=math.sqrt)
		self._half_pi_pulse_func = FunctionWrapper(self._program.applyHalfPiPulse)
		self._pi_pulse_func = FunctionWrapper(self._program.applyPiPulse)

	def _gpu_halfPi(self, a, b):
		self._half_pi_pulse_func(self._env.queue, self._env.constants.ens_shape, a, b)

	def _cpu_halfPi(self, a, b):
		a0 = a.copy()
		b0 = b.copy()
		a[:,:,:] = (a0 - 1j * b0) * math.sqrt(0.5)
		b[:,:,:] = (b0 - 1j * a0) * math.sqrt(0.5)

	def _cpu_halfPiNonIdeal(self, a, b, d_theta, d_phi):
		a0 = a.copy()
		b0 = b.copy()

		half_theta = (math.pi / 2.0 + d_theta) / 2.0
		k1 = numpy.cast[self._env.precision.scalar](math.cos(half_theta))
		k2 = numpy.cast[self._env.precision.complex](-1j * numpy.exp(-1j * d_phi) * math.sin(half_theta))
		k3 = numpy.cast[self._env.precision.complex](-1j * numpy.exp(1j * d_phi) * math.sin(half_theta))

		a[:,:,:] = a0 * k1 + b0 * k2
		b[:,:,:] = a0 * k3 + b0 * k1


class TwoComponentBEC(PairedCalculation):
	"""
	Calculates evolution of two-component BEC, using split-step propagation
	of paired GPEs.
	"""

	def __init__(self, env, d_theta=0, d_phi=0):
		PairedCalculation.__init__(self, env)
		self._env = env

		self._gs = GPEGroundState(env).create()

		self._plan = createPlan(env, env.constants.nvx, env.constants.nvy, env.constants.nvz)
		self._pulse = Pulse(env)

		# indicates whether current state is in midstep (i.e, right after propagation
		# in x-space and FFT to k-space)
		self._midstep = False

		self._prepare()
		self.reset(d_theta, d_phi)

	def _cpu__prepare(self):
		potentials = getPotentials(self._env)
		kvectors = getKVectors(self._env)

		self._potentials = numpy.empty(self._env.constants.ens_shape, dtype=self._env.precision.complex.dtype)
		self._kvectors = numpy.empty(self._env.constants.ens_shape, dtype=self._env.precision.complex.dtype)

		# copy potentials and kvectors making these matrices have the same size
		# as the many-ensemble state
		# it requires additional memory, but makes other code look simpler
		for e in range(self._env.constants.ensembles):
			start = e * self._env.constants.nvz
			stop = (e + 1) * self._env.constants.nvz

			self._potentials[start:stop,:,:] = potentials
			self._kvectors[start:stop,:,:] = kvectors

	def _gpu__prepare(self):

		self._potentials = getPotentials(self._env)
		self._kvectors = getKVectors(self._env)

		kernels = """
			<%!
				from math import sqrt
			%>

			// Initialize ensembles with steady state + noise for evolution calculation
			__kernel void initializeEnsembles(__global ${p.complex.name} *a, __global ${p.complex.name} *b,
				__global ${p.complex.name} *steady_state, __global ${p.complex.name} *a_randoms,
				__global ${p.complex.name} *b_randoms)
			{
				DEFINE_INDEXES;

				${p.complex.name} steady_val = steady_state[index % ${c.cells}];

				${p.scalar.name} coeff = (${p.scalar.name})${1.0 / sqrt(c.dV)};

				//Initialises a-ensemble amplitudes with vacuum noise
				a[index] = steady_val + complex_mul_scalar(a_randoms[index], ${c.V1} * coeff);

				//Initialises b-ensemble amplitudes with vacuum noise
				b[index] = complex_mul_scalar(b_randoms[index], ${c.V2} * coeff);
			}

			// Propagates state vector in k-space for evolution calculation (i.e., in real time)
			__kernel void propagateKSpaceRealTime(__global ${p.complex.name} *a, __global ${p.complex.name} *b,
				${p.scalar.name} dt, read_only image3d_t kvectors)
			{
				DEFINE_INDEXES;

				${p.scalar.name} kvector = get_float_from_image(kvectors, i, j, k % ${c.nvz});
				${p.scalar.name} prop_angle = kvector * dt / 2;
				${p.complex.name} prop_coeff = ${p.complex.ctr}(native_cos(prop_angle), native_sin(prop_angle));

				${p.complex.name} a0 = a[index];
				${p.complex.name} b0 = b[index];

				a[index] = complex_mul(a0, prop_coeff);
				b[index] = complex_mul(b0, prop_coeff);
			}

			// Propagates state vector in x-space for evolution calculation
			__kernel void propagateXSpaceTwoComponent(__global ${p.complex.name} *aa,
				__global ${p.complex.name} *bb, ${p.scalar.name} dt,
				read_only image3d_t potentials)
			{
				DEFINE_INDEXES;

				${p.scalar.name} V = get_float_from_image(potentials, i, j, k % ${c.nvz});

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
					da = cexp(complex_mul_scalar(pa, (dt / 2)));
					db = cexp(complex_mul_scalar(pb, (dt / 2)));

					//propagate to midpoint using log derivative
					a = complex_mul(a0, da);
					b = complex_mul(b0, db);
				%endfor

				//propagate to endpoint using log derivative
				aa[index] = complex_mul(a, da);
				bb[index] = complex_mul(b, db);
			}
		"""

		self._program = self._env.compileSource(kernels)
		self._init_ensembles_func = FunctionWrapper(self._program.initializeEnsembles)
		self._kpropagate_func = FunctionWrapper(self._program.propagateKSpaceRealTime)
		self._xpropagate_func = FunctionWrapper(self._program.propagateXSpaceTwoComponent)

	def _toKSpace(self):
		self._plan.execute(self._a, batch=self._env.constants.ensembles, inverse=True)
		self._plan.execute(self._b, batch=self._env.constants.ensembles, inverse=True)

	def _toXSpace(self):
		self._plan.execute(self._a, batch=self._env.constants.ensembles)
		self._plan.execute(self._b, batch=self._env.constants.ensembles)

	def _gpu__initEnsembles(self, gs, a_randoms, b_randoms):
		a_randoms_gpu = self._env.allocate(a_randoms.shape, a_randoms.dtype)
		b_randoms_gpu = self._env.allocate(b_randoms.shape, b_randoms.dtype)

		cl.enqueue_write_buffer(self._env.queue, a_randoms_gpu, a_randoms)
		cl.enqueue_write_buffer(self._env.queue, b_randoms_gpu, b_randoms)

		self._init_ensembles_func(self._env.queue, self._env.constants.ens_shape,
			self._a, self._b, gs, a_randoms_gpu, b_randoms_gpu)

	def _cpu__initEnsembles(self, gs, a_randoms, b_randoms):
		coeff = 1.0 / math.sqrt(self._env.constants.dV)
		size = self._env.constants.cells * self._env.constants.ensembles

		for e in range(self._env.constants.ensembles):
			start = e * self._env.constants.nvz
			stop = (e + 1) * self._env.constants.nvz

			self._a[start:stop,:,:] = gs + a_randoms[start:stop,:,:] * coeff * self._env.constants.V1
			self._b[start:stop,:,:] = b_randoms[start:stop,:,:] * coeff * self._env.constants.V2

	def _gpu__kpropagate(self, dt):
		self._kpropagate_func(self._env.queue, self._env.constants.ens_shape,
			self._a, self._b, self._env.precision.scalar.dtype(dt), self._kvectors)

	def _cpu__kpropagate(self, dt):
		kcoeff = numpy.exp(self._kvectors * (1j * dt / 2))
		self._a *= kcoeff
		self._b *= kcoeff

	def _gpu__xpropagate(self, dt):
		self._xpropagate_func(self._env.queue, self._env.constants.ens_shape,
			self._a, self._b, self._env.precision.scalar.dtype(dt), self._potentials)

	def _cpu__xpropagate(self, dt):
		a0 = self._a.copy()
		b0 = self._b.copy()

		for iter in xrange(self._env.constants.itmax):
			n_a = numpy.abs(self._a)
			n_b = numpy.abs(self._b)

			n_a = n_a * n_a
			n_b = n_b * n_b

			pa = n_a * n_a * (-self._env.constants.l111 / 2) + n_b * (-self._env.constants.l12 / 2) + \
				1j * (self._potentials + n_a * self._env.constants.g11 + n_b * self._env.constants.g12)

			pb = n_b * (-self._env.constants.l22 / 2) + n_a * (-self._env.constants.l12 / 2) + \
				1j * (self._potentials + n_b * self._env.constants.g22 + n_a * self._env.constants.g12 - self._env.constants.detuning)

			da = numpy.exp(pa * (dt / 2))
			db = numpy.exp(pb * (dt / 2))

			self._a = a0 * da
			self._b = b0 * db

		self._a *= da
		self._b *= db

	def _finishStep(self, dt):
		if self._midstep:
			self._kpropagate(dt)
			self._midstep = False

	def reset(self, d_theta, d_phi):

		self._a = self._env.allocate(self._env.constants.ens_shape, self._env.precision.complex.dtype)
		self._b = self._env.allocate(self._env.constants.ens_shape, self._env.precision.complex.dtype)

		a_randoms = (numpy.random.normal(scale=0.5, size=self._env.constants.ens_shape) +
			1j * numpy.random.normal(scale=0.5, size=self._env.constants.ens_shape)).astype(self._env.precision.complex.dtype)
		b_randoms = (numpy.random.normal(scale=0.5, size=self._env.constants.ens_shape) +
			1j * numpy.random.normal(scale=0.5, size=self._env.constants.ens_shape)).astype(self._env.precision.complex.dtype)

		self._initEnsembles(self._gs, a_randoms, b_randoms)

		# equilibration
		self._toKSpace()
		if self._env.constants.t_equilib > 0:
			for t in xrange(0, self._env.constants.t_equilib, self._env.constants.dt_evo):
				self.propagate(self._env.constants.dt_evo)

		self._t = 0

		# first pi/2 pulse
		# can be done both in x-space and in k-space, because
		# it is a linear transformation
		self._pulse.halfPiNonIdeal(self._a, self._b, d_theta, d_phi)

	def propagate(self, dt):

		# replace two dt/2 k-space propagation by one dt propagation,
		# if there were no rendering between them
		if self._midstep:
			self._kpropagate(dt * 2)
		else:
			self._kpropagate(dt)

		self._toXSpace()
		self._xpropagate(dt)

		self._midstep = True
		self._toKSpace()

		self._t += dt

	def _runCallbacks(self, callbacks):
		self._finishStep(self._env.constants.dt_evo)
		self._toXSpace()
		for callback in callbacks:
			callback(self._t * self._env.constants.t_rho, self._a, self._b)
		self._toKSpace()

	def runEvolution(self, tstop, callbacks, callback_dt=0):
		self._t = 0
		callback_t = 0

		self._runCallbacks(callbacks)

		while self._t * self._env.constants.t_rho < tstop:
			self.propagate(self._env.constants.dt_evo)
			callback_t += self._env.constants.dt_evo * self._env.constants.t_rho

			if callback_t > callback_dt:
				self._runCallbacks(callbacks)
				callback_t = 0

		if callback_dt > tstop:
			self._runCallbacks(callbacks)
