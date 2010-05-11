import numpy
import math

from globals import *
from state import ParticleStatistics, Projection, BlochSphereProjection, Slice
from reduce import getReduce
from evolution import Pulse, TerminateEvolution


class ParticleNumberCollector:

	def __init__(self, env, constants, verbose=False, do_pulse=True):
		self.stats = ParticleStatistics(env, constants)
		self.verbose = verbose
		self._pulse = Pulse(env, constants)
		self._do_pulse = do_pulse

		self.times = []
		self.Na = []
		self.Nb = []

	def __call__(self, t, cloud):
		cloud = cloud.copy()

		if self._do_pulse:
			self._pulse.halfPi(cloud)

		Na = self.stats.countParticles(cloud.a)
		Nb = self.stats.countParticles(cloud.b)
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

	def __init__(self, env, constants, verbose=False, do_pulse=True):
		self.stats = ParticleStatistics(env, constants)
		self.verbose = verbose
		self._pulse = Pulse(env, constants)
		self._do_pulse = do_pulse

		self.previous_Na = None
		self.previous_half = None

	def __call__(self, t, cloud):
		cloud = cloud.copy()

		if self._do_pulse:
			self._pulse.halfPi(cloud)

		Na = self.stats.countParticles(cloud.a)
		Nb = self.stats.countParticles(cloud.b)
		print t, Na, Nb
		half = (Na + Nb) / 2

		if self.previous_Na is None:
			self.previous_Na = Na

		if self.previous_half is None:
			self.previous_half = half

		if (Na > half and self.previous_Na < self.previous_half) or \
				(Na < half and self.previous_Na > self.previous_half):
			raise TerminateEvolution()

class VisibilityCollector:

	def __init__(self, env, constants, verbose=False):
		self.stats = ParticleStatistics(env, constants)
		self.verbose = verbose

		self.times = []
		self.visibility = []

	def __call__(self, t, cloud):
		v = self.stats.getVisibility(cloud.a, cloud.b)

		if self.verbose:
			print "Visibility: " + str((t, v))

		self.times.append(t)
		self.visibility.append(v)

	def getData(self):
		return numpy.array(self.times), numpy.array(self.visibility)


class SurfaceProjectionCollector:

	def __init__(self, env, constants, do_pulse=True):
		self._projection = Projection(env, constants)
		self._pulse = Pulse(env, constants)
		self._do_pulse = do_pulse
		self._constants = constants

		self.times = []
		self.a_xy = []
		self.a_yz = []
		self.b_xy = []
		self.b_yz = []

	def __call__(self, t, cloud):
		"""Returns numbers in units (particles per square micrometer)"""

		cloud = cloud.copy()

		if self._do_pulse:
			self._pulse.halfPi(cloud)

		self.times.append(t)

		# cast density to SI ($mu$m^2 instead of m^2, for better readability)
		coeff_xy = self._constants.dz / (self._constants.l_rho ** 2) / 1e12
		coeff_yz = self._constants.dx / (self._constants.l_rho ** 2) / 1e12

		self.a_xy.append(self._projection.getXY(cloud.a) * coeff_xy)
		self.a_yz.append(self._projection.getYZ(cloud.a) * coeff_yz)
		self.b_xy.append(self._projection.getXY(cloud.b) * coeff_xy)
		self.b_yz.append(self._projection.getYZ(cloud.b) * coeff_yz)

	def getData(self):
		return self.times, self.a_xy, self.a_yz, self.b_xy, self.b_yz


class SliceCollector:

	def __init__(self, env, constants, do_pulse=True):
		self._slice = Slice(env, constants)
		self._pulse = Pulse(env, constants)
		self._do_pulse = do_pulse

		self.times = []
		self.a_xy = []
		self.a_yz = []
		self.b_xy = []
		self.b_yz = []

	def __call__(self, t, cloud):
		"""Returns numbers in units (particles per square micrometer)"""

		cloud = cloud.copy()

		if self._do_pulse:
			self._pulse.halfPi(cloud)

		self.times.append(t)

		# cast density to SI ($mu$m^2 instead of m^2, for better readability)
		coeff_xy = 1.0 / (self._constants.l_rho ** 2) / 1e12
		coeff_yz = 1.0 / (self._constants.l_rho ** 2) / 1e12

		self.a_xy.append(self._slice.getXY(cloud.a) * coeff_xy)
		self.a_yz.append(self._slice.getYZ(cloud.a) * coeff_yz)
		self.b_xy.append(self._slice.getXY(cloud.b) * coeff_xy)
		self.b_yz.append(self._slice.getYZ(cloud.b) * coeff_yz)

	def getData(self):
		return self.times, self.a_xy, self.a_yz, self.b_xy, self.b_yz


class AxialProjectionCollector:

	def __init__(self, env, constants, do_pulse=True):
		self._projection = Projection(env, constants)
		self._pulse = Pulse(env, constants)
		self._do_pulse = do_pulse

		self.times = []
		self.snapshots = []

	def __call__(self, t, cloud):

		cloud = cloud.copy()

		if self._do_pulse:
			self._pulse.halfPi(cloud)

		self.times.append(t)

		a_proj = self._projection.getZ(cloud.a)
		b_proj = self._projection.getZ(cloud.b)

		self.snapshots.append((a_proj - b_proj) / (a_proj + b_proj))

	def getData(self):
		return numpy.array(self.times), numpy.concatenate(self.snapshots).reshape(len(self.times), self.snapshots[0].size).transpose()


class BlochSphereCollector:

	def __init__(self, env, constants, amp_points=64, phase_points=128, amp_range=(0, math.pi),
			phase_range=(0, math.pi * 2)):
		self._bs = BlochSphereProjection(env, constants)
		self._amp_points = amp_points
		self._phase_points = phase_points
		self._amp_range = amp_range
		self._phase_range = phase_range

		self.times = []
		self.snapshots = []

	def __call__(self, t, cloud):
		res = self._bs.getProjection(cloud.a, cloud.b, self._amp_points, self._phase_points, self._amp_range, self._phase_range)

		self.times.append(t)
		self.snapshots.append(res)

	def getData(self):
		return self.times, self.snapshots


class BlochSphereAveragesCollector:

	def __init__(self, env, constants):
		self._bs = BlochSphereProjection(env, constants)

		self.times = []
		self.avg_amps = []
		self.avg_phases = []

	def __call__(self, t, cloud):
		avg_amp, avg_phase = self._bs.getAverages(cloud.a, cloud.b)

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
