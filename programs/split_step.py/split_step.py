import math
import matplotlib
import numpy
import time

try:
	import pycuda.autoinit
	import pycuda.driver as cuda
except:
	pass

from globals import *
from model import Model
from constants import Constants
from ground_state import GPEGroundState
from evolution import TwoComponentBEC
from meters import ParticleStatistics
import typenames

class ParticleNumberPlotter(PairedCalculation):

	def __init__(self, gpu, precision, constants, mempool):
		PairedCalculation.__init__(self, gpu, mempool)
		self.stats = ParticleStatistics(gpu, precision, constants, mempool)
		self.initialN = constants.N
		self.N = constants.N

	def __call__(self, t, a, b):
		Na = self.stats.countParticles(a)
		Nb = self.stats.countParticles(b)
		print t, Na, Nb
		self.N = Na + Nb

	def showLoss(self):
		print "Particle loss: " + str((self.initialN - self.N) / self.initialN * 100) + "%"

class VisibilityPlotter(PairedCalculation):

	def __init__(self, gpu, precision, constants, mempool):
		PairedCalculation.__init__(self, gpu, mempool)
		self.stats = ParticleStatistics(gpu, precision, constants, mempool)
		self.data = []

	def __call__(self, t, a, b):
		v = self.stats.getVisibility(a, b)
		self.data.append((t, v))

	def getData(self):
		return self.data


precision = typenames.single_precision
mempool = GPUPool()
gpu = True

tests = (
	(1, 16, 4e-5),
	(4, 16, 4e-5),
#	(4, 16, 1e-5),
#	(4, 16, 8e-6),
#	(4, 16, 4e-6),
)

results = []

for ensembles, points, dt_evo in tests:
	m = Model()
	m.nvx = points
	m.nvy = points
	m.ensembles = ensembles
	m.dt_evo = dt_evo

	if ensembles == 1:
		m.V1 = 0
		m.V2 = 0

	constants = Constants(m)

	print "--- " + str(constants.shape) + ", " + str(ensembles) + " ensembles, " + \
		str(dt_evo * 1e3) + " ms timestep"

	bec = TwoComponentBEC(gpu, precision, constants, mempool)

	pnumber = ParticleNumberPlotter(gpu, precision, constants, mempool)
	vplotter = VisibilityPlotter(gpu, precision, constants, mempool)

	t1 = time.time()
	bec.runEvolution(0.6, [vplotter], callback_dt=0.01)
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	results.append(vplotter.getData())


for i in range(len(results[0])):
	t = results[0][i][0]

	points = []
	for r in range(len(results)):
		points.append(results[r][i][1])

	print t, " ".join([str(x) for x in points])
