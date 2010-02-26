import math
import matplotlib
import numpy
import time

try:
	import pycuda.autoinit
	import pycuda.driver as cuda
	mempool = GPUPool()
except:
	pass

import matplotlib.pyplot as plt

from globals import *
from config import Model, precision
from constants import Constants
from ground_state import GPEGroundState, ParticleStatistics
from evolution import TwoComponentBEC

constants = Constants(Model)
mempool = GPUPool(stub=True)
gpu = True

bec = TwoComponentBEC(gpu, precision, constants, mempool)

class ParticleNumberPlotter(PairedCalculation):

	def __init__(self, gpu, precision, constants, mempool):
		PairedCalculation.__init__(self, gpu)
		self.stats = ParticleStatistics(gpu, precision, constants, mempool)

	def __call__(self, t, a, b):
		Na = self.stats.countParticles(a)
		Nb = self.stats.countParticles(b)
		print t, Na + Nb

pnumber = ParticleNumberPlotter(gpu, precision, constants, mempool)
bec.runEvolution(0.3, [pnumber], callback_dt=0.005)
