import numpy

from globals import *
from meters import ParticleStatistics, Projection
from reduce import getReduce
from evolution import Pulse


class ParticleNumberCollector:

	def __init__(self, env, verbose=False):
		self._env = env
		self.stats = ParticleStatistics(env)
		self.initialN = env.constants.N
		self.verbose = verbose
		self._pulse = Pulse(env)

		self.times = []
		self.Na = []
		self.Nb = []

	def __call__(self, t, a, b):
		a = self._env.copyBuffer(a)
		b = self._env.copyBuffer(b)

		self._pulse.halfPi(a, b)

		Na = self.stats.countParticles(a)
		Nb = self.stats.countParticles(b)
		if self.verbose:
			print "Particle counter: " + str((t, Na, Nb))

		self.times.append(t)
		self.Na.append(Na)
		self.Nb.append(Nb)

	def getData(self):
		Na = numpy.array(self.Na)
		Nb = numpy.array(self.Nb)
		return numpy.array(self.times), Na, Nb, Na + Nb


class VisibilityCollector:

	def __init__(self, env, verbose=False):
		self.stats = ParticleStatistics(env)
		self.verbose = verbose

		self.times = []
		self.visibility = []

	def __call__(self, t, a, b):
		v = self.stats.getVisibility(a, b)

		if self.verbose:
			print "Visibility: " + str((t, v))

		self.times.append(t)
		self.visibility.append(v)

	def getData(self):
		return numpy.array(self.times), numpy.array(self.visibility)


class SurfaceProjectionCollector:

	def __init__(self, env):
		self._env = env
		self._projection = Projection(env)
		self._pulse = Pulse(env)

		self.times = []
		self.a_xy = []
		self.a_yz = []
		self.b_xy = []
		self.b_yz = []

	def __call__(self, t, a, b):
		"""Returns numbers in units (particles per square micrometer)"""

		a = self._env.copyBuffer(a)
		b = self._env.copyBuffer(b)
		self._pulse.halfPi(a, b)

		self.times.append(t)

		# cast density to SI ($mu$m^2 instead of m^2, for better readability)
		coeff_xy = self._env.constants.dz / (self._env.constants.l_rho ** 2) / 1e12
		coeff_yz = self._env.constants.dx / (self._env.constants.l_rho ** 2) / 1e12

		self.a_xy.append(self._projection.getXY(a) * coeff_xy)
		self.a_yz.append(self._projection.getYZ(a) * coeff_yz)
		self.b_xy.append(self._projection.getXY(b) * coeff_xy)
		self.b_yz.append(self._projection.getYZ(b) * coeff_yz)

	def getData(self):
		return self.times, self.a_xy, self.a_yz, self.b_xy, self.b_yz


class AxialProjectionCollector:

	def __init__(self, env):
		self._env = env
		self._projection = Projection(env)
		self._pulse = Pulse(env)

		self.times = []
		self.snapshots = []

	def __call__(self, t, a, b):

		a = self._env.copyBuffer(a)
		b = self._env.copyBuffer(b)
		self._pulse.halfPi(a, b)

		self.times.append(t)

		a_proj = self._projection.getZ(a)
		b_proj = self._projection.getZ(b)

		self.snapshots.append((a_proj - b_proj) / (a_proj + b_proj))

	def getData(self):
		return numpy.array(self.times), numpy.concatenate(self.snapshots).reshape(len(self.times), self.snapshots[0].size)
