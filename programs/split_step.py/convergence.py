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

class VisibilityPlotter(PairedCalculation):

	def __init__(self, gpu, precision, constants, mempool):
		PairedCalculation.__init__(self, gpu, mempool)
		self.stats = ParticleStatistics(gpu, precision, constants, mempool)

	def __call__(self, t, a, b):
		self.v = self.stats.getVisibility(a, b)

	def getData(self):
		return self.v


precision = typenames.single_precision
mempool = GPUPool()
gpu = False

def getVisibility(dt, points, ensembles):
	m = Model()
	m.nvx = points
	m.nvy = points
	m.ensembles = ensembles
	m.dt_evo = dt

	if ensembles == 1:
		m.V1 = 0
		m.V2 = 0

	constants = Constants(m)

	bec = TwoComponentBEC(gpu, precision, constants, mempool)

	vplotter = VisibilityPlotter(gpu, precision, constants, mempool)

	bec.runEvolution(0.05, [vplotter], callback_dt=1)

	return vplotter.getData()

print getVisibility(1e-4, 16, 1)
