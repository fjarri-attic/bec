import numpy
import math

from globals import *
from meters import ParticleStatistics, Projection, BlochSphereProjection, Slice
from reduce import getReduce
from evolution import Pulse, TerminateEvolution


class ParticleNumberCollector:

	def __init__(self, env, verbose=False, do_pulse=True):
		self._env = env
		self.stats = ParticleStatistics(env)
		self.initialN = env.constants.N
		self.verbose = verbose
		self._pulse = Pulse(env)
		self._do_pulse = do_pulse

		self.times = []
		self.Na = []
		self.Nb = []

	def __call__(self, t, a, b):
		a = self._env.copyBuffer(a)
		b = self._env.copyBuffer(b)

		if self._do_pulse:
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


class EqualParticleNumberCondition:

	def __init__(self, env, verbose=False):
		self._env = env
		self.stats = ParticleStatistics(env)
		self.verbose = verbose
		self._pulse = Pulse(env)

		self.previous_Na = None
		self.previous_half = None

	def __call__(self, t, a, b):
		a = self._env.copyBuffer(a)
		b = self._env.copyBuffer(b)

		self._pulse.halfPi(a, b)

		Na = self.stats.countParticles(a)
		Nb = self.stats.countParticles(b)
		half = (Na + Nb) / 2

		if self.previous_Na is None:
			self.previous_Na = Na

		if self.previous_half is None:
			self.previous_half = half

		if (Na > half and self.previous_Na < self.previous_half) or \
				(Na < half and self.previous_Na > self.previous_half):
			raise TerminateEvolution()

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


class SliceCollector:

	def __init__(self, env):
		self._env = env
		self._slice = Slice(env)
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
		coeff_xy = 1.0 / (self._env.constants.l_rho ** 2) / 1e12
		coeff_yz = 1.0 / (self._env.constants.l_rho ** 2) / 1e12

		self.a_xy.append(self._slice.getXY(a) * coeff_xy)
		self.a_yz.append(self._slice.getYZ(a) * coeff_yz)
		self.b_xy.append(self._slice.getXY(b) * coeff_xy)
		self.b_yz.append(self._slice.getYZ(b) * coeff_yz)

	def getData(self):
		return self.times, self.a_xy, self.a_yz, self.b_xy, self.b_yz


class AxialProjectionCollector:

	def __init__(self, env, do_pulse=True):
		self._env = env
		self._projection = Projection(env)
		self._pulse = Pulse(env)
		self._do_pulse = do_pulse

		self.times = []
		self.snapshots = []

	def __call__(self, t, a, b):

		a = self._env.copyBuffer(a)
		b = self._env.copyBuffer(b)

		if self._do_pulse:
			self._pulse.halfPi(a, b)

		self.times.append(t)

		a_proj = self._projection.getZ(a)
		b_proj = self._projection.getZ(b)

		self.snapshots.append((a_proj - b_proj) / (a_proj + b_proj))

	def getData(self):
		return numpy.array(self.times), numpy.concatenate(self.snapshots).reshape(len(self.times), self.snapshots[0].size).transpose()


class BlochSphereCollector:

	def __init__(self, env, amp_points=64, phase_points=128, amp_range=(0, math.pi),
			phase_range=(0, math.pi * 2)):
		self._env = env
		self._bs = BlochSphereProjection(env)
		self._amp_points = amp_points
		self._phase_points = phase_points
		self._amp_range = amp_range
		self._phase_range = phase_range

		self.times = []
		self.snapshots = []

	def __call__(self, t, a, b):
		res = self._bs.getProjection(a, b, self._amp_points, self._phase_points, self._amp_range, self._phase_range)

		self.times.append(t)
		self.snapshots.append(res)

	def getData(self):
		return self.times, self.snapshots


class BlochSphereAveragesCollector:

	def __init__(self, env):
		self._env = env
		self._bs = BlochSphereProjection(env)

		self.times = []
		self.avg_amps = []
		self.avg_phases = []

	def __call__(self, t, a, b):
		avg_amp, avg_phase = self._bs.getAverages(a, b)

		self.times.append(t)
		self.avg_amps.append(avg_amp)
		self.avg_phases.append(avg_phase)

	def getData(self):
		return self.times, self.avg_amps, self.avg_phases

	@staticmethod
	def getSnapshots(collectors):
		snapshots_num = len(collectors[0].avg_amps)
		points_num = len(collectors)

		amp_min, amp_max = 0.0, math.pi
		phase_min, phase_max = 0.0, 2.0 * math.pi
		amp_points = 64
		phase_points = 128

		d_amp = (amp_max - amp_min) / (amp_points - 1)
		d_phase = (phase_max - phase_min) / (phase_points - 1)

		res = []
		for i in xrange(snapshots_num):
			snapshot = numpy.zeros((amp_points, phase_points))

			for j in xrange(points_num):
				amp = collectors[j].avg_amps[i]
				phase = collectors[j].avg_phases[i]

				snapshot[int(amp / d_amp), int(phase / d_phase)] += 1

			res.append(snapshot)

		return res
