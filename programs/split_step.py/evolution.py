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
from constants import PSI_FUNC, WIGNER

from decompose import getTakagiDecomposition


class TerminateEvolution(Exception):
	pass


class Pulse(PairedCalculation):

	def __init__(self, env, constants):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants

		self._plan = createPlan(env, constants, constants.nvx, constants.nvy, constants.nvz)

		self._potentials = getPotentials(env, constants)
		self._kvectors = getKVectors(env, constants)

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _gpu__prepare(self):
		kernels = """
			<%!
				from math import sqrt
			%>

			// Apply pi/2 pulse (instantaneus approximation)
			__kernel void applyHalfPi(__global ${c.complex.name} *a, __global ${c.complex.name} *b)
			{
				DEFINE_INDEXES;

				${c.complex.name} a0 = a[index];
				${c.complex.name} b0 = b[index];

				${c.complex.name} minus_i = ${c.complex.ctr}(0, -1);
				a[index] = complex_mul_scalar(a0 + complex_mul(b0, minus_i), (${c.scalar.name})${sqrt(0.5)});
				b[index] = complex_mul_scalar(complex_mul(a0, minus_i) + b0, (${c.scalar.name})${sqrt(0.5)});
			}

			// Apply pi pulse (instantaneus approximation)
			__kernel void applyPi(__global ${c.complex.name} *a, __global ${c.complex.name} *b)
			{
				DEFINE_INDEXES;

				${c.complex.name} a0 = a[index];
				${c.complex.name} b0 = b[index];

				${c.complex.name} minus_i = ${c.complex.ctr}(0, -1);

				a[index] = complex_mul(b0, minus_i);
				b[index] = complex_mul(a0, minus_i);
			}

			__kernel void apply(__global ${c.complex.name} *a, __global ${c.complex.name} *b,
				${c.scalar.name} theta, ${c.scalar.name} phi)
			{
				DEFINE_INDEXES;

				${c.complex.name} a0 = a[index];
				${c.complex.name} b0 = b[index];

				${c.scalar.name} sin_half_theta = sin(theta / 2);
				${c.scalar.name} cos_half_theta = cos(theta / 2);

				${c.complex.name} minus_i = ${c.complex.ctr}(0, -1);

				${c.complex.name} k2 = complex_mul_scalar(complex_mul(
					minus_i, cexp(${c.complex.ctr}(0, -phi))
				), sin_half_theta);

				${c.complex.name} k3 = complex_mul_scalar(complex_mul(
					minus_i, cexp(${c.complex.ctr}(0, phi))
				), sin_half_theta);

				a[index] = complex_mul_scalar(a0, cos_half_theta) + complex_mul(b0, k2);
				b[index] = complex_mul_scalar(b0, cos_half_theta) + complex_mul(a0, k3);
			}
		"""

		self._program = self._env.compile(kernels, self._constants)
		self._half_pi_func = self._program.applyHalfPi
		self._pi_func = self._program.applyPi
		self._func = self._program.apply

	def _gpu_halfPi(self, cloud):
		self._half_pi_func(cloud.a.shape, cloud.a.data, cloud.b.data)

	def _cpu_halfPi(self, cloud):
		a = cloud.a.data
		b = cloud.b.data
		a0 = a.copy()
		b0 = b.copy()
		a[:,:,:] = (a0 - 1j * b0) * math.sqrt(0.5)
		b[:,:,:] = (b0 - 1j * a0) * math.sqrt(0.5)

	def _cpu_apply(self, cloud, theta, phi):
		a = cloud.a.data
		b = cloud.b.data

		a0 = a.copy()
		b0 = b.copy()

		half_theta = theta / 2.0
		k1 = self._constants.scalar.cast(math.cos(half_theta))
		k2 = self._constants.complex.cast(-1j * numpy.exp(-1j * phi) * math.sin(half_theta))
		k3 = self._constants.complex.cast(-1j * numpy.exp(1j * phi) * math.sin(half_theta))

		a[:,:,:] = a0 * k1 + b0 * k2
		b[:,:,:] = a0 * k3 + b0 * k1

	def _gpu_apply(self, cloud, theta, phi):
		self._func(cloud.a.shape, cloud.a.data, cloud.b.data,
			self._constants.scalar.cast(theta),
			self._constants.scalar.cast(phi))

	def _cpu__propagateRK(self, cloud, t, dt):

		shape = cloud.a.shape
		dtype = cloud.a.dtype
		batch = cloud.a.size / self._constants.cells
		nvz = self._constants.nvz

		comp1 = cloud.a.comp
		comp2 = cloud.b.comp
		g = self._constants.g
		g11 = g[(comp1, comp1)]
		g12 = g[(comp1, comp2)]
		g22 = g[(comp2, comp2)]

		l111 = self._constants.l111
		l12 = self._constants.l12
		l22 = self._constants.l22

		def func(a_data, b_data):
			n_a = numpy.abs(a_data) ** 2
			n_b = numpy.abs(b_data) ** 2

			a_kdata = self._env.allocate(shape, dtype=dtype)
			b_kdata = self._env.allocate(shape, dtype=dtype)
			a_res = self._env.allocate(shape, dtype=dtype)
			b_res = self._env.allocate(shape, dtype=dtype)

			self._plan.execute(a_data, a_kdata, inverse=True, batch=batch)
			self._plan.execute(b_data, b_kdata, inverse=True, batch=batch)

			for e in xrange(batch):
				start = e * nvz
				stop = (e + 1) * nvz
				a_res[start:stop,:,:] = 1j * (a_kdata[start:stop,:,:] * self._kvectors -
					a_data[start:stop,:,:] * self._potentials)
				b_res[start:stop,:,:] = 1j * (b_kdata[start:stop,:,:] * self._kvectors -
					b_data[start:stop,:,:] * self._potentials)

			a_res += (
				-(n_a * n_a * (l111 / 2) + n_b * (l12 / 2)) +
				1j * (-n_a * g11 - n_b * g12)) * a_data - \
				0.5j * self._constants.rabi_freq * \
					numpy.exp(-1j * t * self._constants.detuning) * b_data

			b_res += (
				-(n_b * (l22 / 2) + n_a * (l12 / 2)) +
				1j * (-n_b * g22 - n_a * g12)) * b_data - \
				0.5j * self._constants.rabi_freq * \
					numpy.exp(1j * t * self._constants.detuning) * a_data

			return a_res, b_res

		a0 = cloud.a.data.copy()
		b0 = cloud.b.data.copy()

		a_k1, b_k1 = func(a0, b0)
		a_k2, b_k2 = func(a0 + 0.5 * dt * a_k1, b0 + 0.5 * dt * b_k1)
		a_k3, b_k3 = func(a0 + 0.5 * dt * a_k2, b0 + 0.5 * dt * b_k2)
		a_k4, b_k4 = func(a0 + dt * a_k3, b0 + dt * b_k3)

		cloud.a.data += 1.0 / 6.0 * dt * (a_k1 + 2.0 * a_k2 + 2.0 * a_k3 + a_k4)
		cloud.b.data += 1.0 / 6.0 * dt * (b_k1 + 2.0 * b_k2 + 2.0 * b_k3 + b_k4)

	def applyNonIdeal(self, cloud, theta):
		tmax = theta * self._constants.rabi_period
		steps = 100
		dt = tmax / steps

		t = 0.0
		for i in xrange(steps):
			t += dt
			print numpy.sum(numpy.abs(cloud.a.data) ** 2 * self._constants.dV), \
				numpy.sum(numpy.abs(cloud.b.data) ** 2 * self._constants.dV)
			self._propagateRK(cloud, t, dt)


class TwoComponentEvolution(PairedCalculation):
	"""
	Calculates evolution of two-component BEC, using split-step propagation
	of paired GPEs.
	"""

	def __init__(self, env, constants):
		PairedCalculation.__init__(self, env)
		self._env = env
		self._constants = constants

		self._plan = createPlan(env, constants, constants.nvx, constants.nvy, constants.nvz)
		self._pulse = Pulse(env, constants)

		# indicates whether current state is in midstep (i.e, right after propagation
		# in x-space and FFT to k-space)
		self._midstep = False

		self._potentials = getPotentials(self._env, self._constants)
		self._kvectors = getKVectors(self._env, self._constants)

		self._prepare()

	def _cpu__prepare(self):
		pass

	def _gpu__prepare(self):

		kernels = """
			<%!
				from math import sqrt
			%>

			// Propagates state vector in k-space for evolution calculation (i.e., in real time)
			__kernel void propagateKSpaceRealTime(__global ${c.complex.name} *a, __global ${c.complex.name} *b,
				${c.scalar.name} dt, read_only image3d_t kvectors)
			{
				DEFINE_INDEXES;

				${c.scalar.name} kvector = get_float_from_image(kvectors, i, j, k % ${c.nvz});
				${c.scalar.name} prop_angle = kvector * dt / 2;
				${c.complex.name} prop_coeff = ${c.complex.ctr}(native_cos(prop_angle), native_sin(prop_angle));

				${c.complex.name} a0 = a[index];
				${c.complex.name} b0 = b[index];

				a[index] = complex_mul(a0, prop_coeff);
				b[index] = complex_mul(b0, prop_coeff);
			}

			// Propagates state vector in x-space for evolution calculation
			%for suffix in ('', 'Wigner'):
			__kernel void propagateXSpaceTwoComponent${suffix}(__global ${c.complex.name} *aa,
				__global ${c.complex.name} *bb, ${c.scalar.name} dt,
				read_only image3d_t potentials)
			{
				DEFINE_INDEXES;

				${c.scalar.name} V = get_float_from_image(potentials, i, j, k % ${c.nvz});

				${c.complex.name} a = aa[index];
				${c.complex.name} b = bb[index];

				//store initial x-space field
				${c.complex.name} a0 = a;
				${c.complex.name} b0 = b;

				${c.complex.name} pa, pb, da = ${c.complex.ctr}(0, 0), db = ${c.complex.ctr}(0, 0);
				${c.scalar.name} n_a, n_b;

				//iterate to midpoint solution
				%for iter in range(c.itmax):
					n_a = squared_abs(a);
					n_b = squared_abs(b);

					// TODO: there must be no minus sign before imaginary part,
					// but without it the whole thing diverges
					pa = ${c.complex.ctr}(
						-(${c.l111} * n_a * n_a + ${c.l12} * n_b) / 2,
						-(-V - ${c.g11} * n_a - ${c.g12} * n_b));
					pb = ${c.complex.ctr}(
						-(${c.l22} * n_b + ${c.l12} * n_a) / 2,
						-(-V - ${c.g22} * n_b - ${c.g12} * n_a + ${c.detuning}));

					%if suffix == "Wigner":
						pa += ${c.complex.ctr}(
							(1.5 * n_a - 0.75 / ${c.dV}) * ${c.l111} + ${c.l12} * 0.25,
							-(${c.g11} + 0.5 * ${c.g12})) / ${c.dV};
						pb += ${c.complex.ctr}(
							${c.l12} * 0.25 + ${c.l22} * 0.5,
							-(${c.g22} + 0.5 * ${c.g12})) / ${c.dV};
					%endif

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
			%endfor
		"""

		self._program = self._env.compile(kernels, self._constants)
		self._kpropagate_func = self._program.propagateKSpaceRealTime
		self._xpropagate_func = self._program.propagateXSpaceTwoComponent
		self._xpropagate_wigner = self._program.propagateXSpaceTwoComponentWigner

	def _toKSpace(self, cloud):
		batch = cloud.a.size / self._constants.cells
		self._plan.execute(cloud.a.data, batch=batch, inverse=True)
		self._plan.execute(cloud.b.data, batch=batch, inverse=True)

	def _toXSpace(self, cloud):
		batch = cloud.a.size / self._constants.cells
		self._plan.execute(cloud.a.data, batch=batch)
		self._plan.execute(cloud.b.data, batch=batch)

	def _gpu__kpropagate(self, cloud, dt):
		self._kpropagate_func(cloud.a.shape,
			cloud.a.data, cloud.b.data, self._constants.scalar.cast(dt), self._kvectors)

	def _cpu__kpropagate(self, cloud, dt):
		kcoeff = numpy.exp(self._kvectors * (1j * dt / 2))
		data1 = cloud.a.data
		data2 = cloud.b.data
		nvz = self._constants.nvz

		for e in xrange(cloud.a.size / self._constants.cells):
			start = e * nvz
			stop = (e + 1) * nvz
			data1[start:stop,:,:] *= kcoeff
			data2[start:stop,:,:] *= kcoeff

	def _gpu__xpropagate(self, cloud, dt):
		if cloud.type == WIGNER:
			func = self._xpropagate_wigner
		else:
			func = self._xpropagate_func

		func(cloud.a.shape, cloud.a.data, cloud.b.data,
			self._constants.scalar.cast(dt), self._potentials)

	def _cpu__getNoiseTerms(self, cloud):
		noise_a = numpy.zeros(self._constants.ens_shape, dtype=self._constants.complex.dtype)
		noise_b = numpy.zeros(self._constants.ens_shape, dtype=self._constants.complex.dtype)

		shape = self._constants.ens_shape

		eta = [numpy.random.normal(scale=math.sqrt(self._constants.dt_evo / self._constants.dV),
			size=shape).astype(self._constants.scalar.dtype) for i in xrange(4)]

		n1 = numpy.abs(cloud.a.data) ** 2
		n2 = numpy.abs(cloud.b.data) ** 2
		dV = self._constants.dV
		l12 = self._constants.l12
		l22 = self._constants.l22
		l111 = self._constants.l111

		a = 0.25 * l12 * (n2 - 0.5 / dV) + 0.25 * l111 * (3.0 * n1 * n1 - 6.0 * n1 / dV + 1.5 / dV / dV)
		d = 0.25 * l12 * (n1 - 0.5 / dV) + 0.25 * l22 * (2.0 * n2 - 1.0 / dV)
		t = 0.25 * l12 * cloud.a.data * numpy.conj(cloud.b.data)
		b = numpy.real(t)
		c = numpy.imag(t)

		t1 = numpy.sqrt(a)
		t2 = numpy.sqrt(d - (b ** 2) / a)
		t3 = numpy.sqrt(a + a * (c ** 2) / (b ** 2 - a * d))

		row1 = t1 * eta[0]
		row2 = b / t1 * eta[0] + t2 * eta[1]
		row3 = c / t2 * eta[1] + t3 * eta[2]
		row4 = -c / t1 * eta[0] + b * c / (a * t2) * eta[1] + b / a * t3 * eta[2] + \
			numpy.sqrt((a ** 2 - b ** 2 - c ** 2) / a) * eta[3]

		noise_a = row1 + 1j * row3
		noise_b = row2 + 1j * row4

		noise_a /= (cloud.a.data * self._constants.dt_evo)
		noise_b /= (cloud.b.data * self._constants.dt_evo)

		#print numpy.sum(numpy.abs(noise_a))
		#print numpy.sum(numpy.abs(numpy.nan_to_num(noise_a)))

		return numpy.nan_to_num(noise_a), numpy.nan_to_num(noise_b)

	def _cpu__xpropagate(self, cloud, dt):
		a = cloud.a
		b = cloud.b
		a0 = a.data.copy()
		b0 = b.data.copy()

		comp1 = cloud.a.comp
		comp2 = cloud.b.comp
		g = self._constants.g
		g11 = g[(comp1, comp1)]
		g12 = g[(comp1, comp2)]
		g22 = g[(comp2, comp2)]

		l111 = self._constants.l111
		l12 = self._constants.l12
		l22 = self._constants.l22

		p = self._potentials * 1j
		nvz = self._constants.nvz

		if cloud.type == WIGNER:
			noise_a, noise_b = self._getNoiseTerms(cloud)

		for iter in xrange(self._constants.itmax):
			n_a = numpy.abs(a.data) ** 2
			n_b = numpy.abs(b.data) ** 2

			pa = n_a * n_a * (-l111 / 2) + n_b * (-l12 / 2) + \
				1j * (n_a * g11 + n_b * g12)

			pb = n_b * (-l22 / 2) + n_a * (-l12 / 2) + \
				1j * (n_b * g22 + n_a * g12)

			for e in xrange(cloud.a.size / self._constants.cells):
				start = e * nvz
				stop = (e + 1) * nvz
				pa[start:stop] += p
				pb[start:stop] += p

			if cloud.type == WIGNER:
				dV = self._constants.dV
				pa += ((1.5 * n_a - 0.75 / dV) * l111 + l12 * 0.25 -
					1j * (g11 + 0.5 * g12)) / dV
				pb += (l12 * 0.25 + l22 * 0.5 -
					1j * (g22 + 0.5 * g12)) / dV

				#print numpy.sum(numpy.abs(noise_a)) / numpy.sum(numpy.abs(pa)), \
				#	numpy.sum(numpy.abs(noise_b)) / numpy.sum(numpy.abs(pb))

				pa += noise_a * 1j
				pb += noise_b * 1j

			da = numpy.exp(pa * (dt / 2))
			db = numpy.exp(pb * (dt / 2))

			a.data = a0 * da
			b.data = b0 * db

		a.data *= da
		b.data *= db

	def _finishStep(self, cloud, dt):
		if self._midstep:
			self._kpropagate(cloud, dt)
			self._midstep = False

	def propagate(self, cloud, dt):

		# replace two dt/2 k-space propagation by one dt propagation,
		# if there were no rendering between them
		if self._midstep:
			self._kpropagate(cloud, dt * 2)
		else:
			self._kpropagate(cloud, dt)

		self._toXSpace(cloud)
		self._xpropagate(cloud, dt)

		self._midstep = True
		self._toKSpace(cloud)

	def _runCallbacks(self, t, cloud, callbacks):
		if callbacks is None:
			return

		self._finishStep(cloud, self._constants.dt_evo)
		self._toXSpace(cloud)
		for callback in callbacks:
			callback(t, cloud)
		self._toKSpace(cloud)

	def run(self, cloud, time, callbacks=None, callback_dt=0):

		# in SI units
		t = 0
		callback_t = 0
		t_rho = self._constants.t_rho

		# in natural units
		dt = self._constants.dt_evo

		self._toKSpace(cloud)

		try:
			self._runCallbacks(0, cloud, callbacks)

			while t < time:
				self.propagate(cloud, dt)
				t += dt * t_rho
				callback_t += dt * t_rho

				if callback_t > callback_dt:
					self._runCallbacks(t, cloud, callbacks)
					callback_t = 0

			if callback_dt > time:
				self._runCallbacks(time, cloud, callbacks)

			self._toXSpace(cloud)

		except TerminateEvolution:
			return t
