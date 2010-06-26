import numpy
import time
import math

from globals import Environment
from model import Model
from constants import Constants
from evolution import TwoComponentBEC
import typenames

from collectors import *

from datahelpers import XYData, HeightmapData, XYPlot, HeightmapPlot

m = Model()
constants = Constants(m)
env = Environment(False, typenames.double_precision, constants)
bec = TwoComponentBEC(env)

experiments_num = 64
bas = [BlochSphereAveragesCollector(env) for i in xrange(experiments_num)]

for i in xrange(experiments_num):
	t1 = time.time()
	bec.reset(numpy.random.normal(scale=1.0/math.sqrt(env.constants.N)),
		numpy.random.normal(scale=1.0/math.sqrt(env.constants.N)))
	#bec.reset(0, 0)
	bec.runEvolution(0.01, [bas[i]], callback_dt=0.005)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"
	#print bas[i].avg_amps
	#print bas[i].avg_phases

snapshots = BlochSphereAveragesCollector.getSnapshots(bas)
times = bas[0].times

for t, snapshot in zip(times, snapshots):
	pr = HeightmapData("BS projection test", snapshot, xmin=0, xmax=2 * math.pi,
		ymin=0, ymax=math.pi, zmin=0, xname="Phase", yname="Amplitude", zname="Density")
	pr = HeightmapPlot(pr)
	pr.save('test' + str(int(t * 1000 + 0.5)).zfill(3) + '.png')
