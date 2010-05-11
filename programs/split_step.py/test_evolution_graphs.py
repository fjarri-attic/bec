import numpy
import time
import math

from globals import Environment
from model import Model
from constants import Constants
from evolution import TwoComponentEvolution
from ground_state import GPEGroundState

from collectors import *

from datahelpers import XYData, HeightmapData, XYPlot, HeightmapPlot, EvolutionPlot

# preparation
env = Environment(gpu=False)
constants = Constants(Model(N=150000), double_precision=False)
gs = GPEGroundState(env, constants)
evolution = TwoComponentEvolution(env, constants)
pulse = Pulse(env, constants)
a = SurfaceProjectionCollector(env, constants)

# experiment
cloud = gs.create()
pulse.halfPi(cloud)
t1 = time.time()
evolution.run(cloud, time=0.399, callbacks=[a], callback_dt=0.01)
env.synchronize()
t2 = time.time()
print "Time spent: " + str(t2 - t1) + " s"

# render
times, a_xy, a_yz, b_xy, b_yz = a.getData()

times = [str(int(x * 1000 + 0.5)) for x in times]

for name, dataset in (('testa.pdf', a_yz), ('testb.pdf', b_yz)):
	hms = []
	for t, hm in zip(times, dataset):
		hms.append(HeightmapData(t, hm.transpose(),
			xmin=-constants.zmax, xmax=constants.zmax,
			ymin=-constants.ymax, ymax=constants.ymax,
			zmin=0, zmax=400))

	EvolutionPlot(hms, shape=(8, 5)).save(name)
