import math
import matplotlib
import numpy
import time

try:
	import pycuda.autoinit
	import pycuda.driver as cuda
except:
	pass

import matplotlib.pyplot as plt

from globals import *
from config import Model
from constants import Constants
from ground_state import GPEGroundState, ParticleStatistics
from evolution import TwoComponentBEC
import typenames

class ParticleNumberPlotter(PairedCalculation):

	def __init__(self, gpu, precision, constants, mempool):
		PairedCalculation.__init__(self, gpu)
		self.stats = ParticleStatistics(gpu, precision, constants, mempool)
		self.initialN = constants.N
		self.N = constants.N

	def __call__(self, t, a, b):
		Na = self.stats.countParticles(a)
		Nb = self.stats.countParticles(b)
		self.N = Na + Nb

	def showLoss(self):
		print "Particle loss: " + str((self.initialN - self.N) / self.initialN * 100) + "%"


precision = typenames.single_precision
mempool = GPUPool()
gpu = True

#tests = ((1, 16), (4, 16), (1, 32), (4, 32))
tests = ((1, 16),)

for ensembles, points in tests:
	m = Model()
	m.nvx = points
	m.nvy = points
	m.ensembles = ensembles

	if ensembles == 1:
		m.V1 = 0
		m.V2 = 0

	m.gamma111 = 0
	m.gamma12 = 0
	m.gamma22 = 0

	print "--- " + str(points) + " points, " + str(ensembles) + " ensembles:"

	constants = Constants(m)
	bec = TwoComponentBEC(gpu, precision, constants, mempool)
	pnumber = ParticleNumberPlotter(gpu, precision, constants, mempool)

	t1 = time.time()
	bec.runEvolution(0.3, [pnumber], callback_dt=5)
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	pnumber.showLoss()
