import numpy
import time
import math

from globals import Environment
from model import Model
from constants import Constants
from evolution import TwoComponentBEC
import typenames

from collectors import *

from datahelpers import XYData, HeightmapData, XYPlot, HeightmapPlot, EvolutionPlot

m = Model()
constants = Constants(m)
env = Environment(True, typenames.single_precision, constants)
bec = TwoComponentBEC(env)

a = SurfaceProjectionCollector(env)
t1 = time.time()
bec.runEvolution(0.199, [a], callback_dt=0.01)
env.synchronize()
t2 = time.time()
print "Time spent: " + str(t2 - t1) + " s"

times, a_xy, a_yz, b_xy, b_yz = a.getData()

times = [str(int(x * 1000 + 0.5)) for x in times]

for name, dataset in (('testa.pdf', a_yz), ('testb.pdf', b_yz)):
	hms = []
	for t, hm in zip(times, dataset):
		hms.append(HeightmapData(t, hm.transpose(),
			xmin=-env.constants.zmax, xmax=env.constants.zmax,
			ymin=-env.constants.ymax, ymax=env.constants.ymax,
			zmin=0, zmax=700))

	EvolutionPlot(hms, shape=(5, 4)).save(name)
