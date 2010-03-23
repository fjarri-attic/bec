import math
import matplotlib
import numpy
import time

try:
	import pyopencl as cl
except:
	pass

from globals import *
from model import Model
from constants import Constants
from ground_state import GPEGroundState
#from evolution import TwoComponentBEC
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

	def __call__(self, t, a, b):
		self.v = self.stats.getVisibility(a, b)

	def getData(self):
		return self.v

for gpu in (True, False):
	print "GPU=" + str(gpu)
	env = Environment(gpu, typenames.single_precision, Constants(Model))

	t1 = time.time()
	tf = GPEGroundState(env)
	state = tf.create()

	if env.gpu:
		env.queue.finish()

	t2 = time.time()
	print str(t2 - t1) + " sec"

exit()

tests = (
	(8, 16, 2e-5),
	(8, 16, 3e-5),
	(8, 16, 4e-5),
	(8, 16, 5e-5),
	(8, 16, 6e-5),
	(8, 16, 7e-5),
	(8, 16, 8e-5),
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
	bec.runEvolution(0.05, [vplotter], callback_dt=1)
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	print vplotter.getData()
	results.append(vplotter.getData())

def findConvergence(l):
	pos = 0
	res = 999999
	for i, x in enumerate(l):
		if i == 0:
			diff = abs(x - l[1])
		elif i == len(l) - 1:
			diff = abs(x - l[-2])
		else:
			diff = (abs(x - l[i-1]) + abs(x - l[i+1])) / 2
		if diff < res:
			res = diff
			pos = i

	print pos

print results, findConvergence(results)


#print results
#for i in range(len(results[0])):
#	t = results[0][i][0]
#
#	points = []
#	for r in range(len(results)):
#		points.append(results[r][i][1])
#
#	print t, " ".join([str(x) for x in points])
