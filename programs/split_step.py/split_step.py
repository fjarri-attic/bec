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
from evolution import TwoComponentBEC
from meters import ParticleStatistics
import typenames

class ParticleNumberPlotter(PairedCalculation):

	def __init__(self, env):
		PairedCalculation.__init__(self, env)
		self.stats = ParticleStatistics(env)
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

	def __init__(self, env):
		PairedCalculation.__init__(self, env)
		self.stats = ParticleStatistics(env)

	def __call__(self, t, a, b):
		self.v = self.stats.getVisibility(a, b)

	def getData(self):
		return self.v


for gpu in (True, False):
	m = Model()

	constants = Constants(m)
	env = Environment(gpu, typenames.single_precision, constants)

	print str(env)

	bec = TwoComponentBEC(env)

	vplotter = VisibilityPlotter(env)

	t1 = time.time()
	bec.runEvolution(0.05, [vplotter], callback_dt=1)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	print vplotter.getData()
