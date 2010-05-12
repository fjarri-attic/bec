import numpy
import time
import math

from globals import Environment
from model import Model
from constants import Constants
from evolution import TwoComponentEvolution
from ground_state import GPEGroundState

from collectors import *

from datahelpers import XYData, HeightmapData, XYPlot, HeightmapPlot

# preparation
env = Environment(gpu=False)
constants = Constants(Model(N=150000), double_precision=False)
gs = GPEGroundState(env, constants)
evolution = TwoComponentEvolution(env, constants)
pulse = Pulse(env, constants)
a = AxialProjectionCollector(env, constants)

# experiment
cloud = gs.createCloud()
pulse.halfPi(cloud)
t1 = time.time()
evolution.run(cloud, time=0.3, callbacks=[a], callback_dt=0.005)
env.synchronize()
t2 = time.time()
print "Time spent: " + str(t2 - t1) + " s"

times, picture = a.getData()

pr = HeightmapData("test", picture, xmin=0, xmax=300,
	ymin=-constants.zmax, ymax=constants.zmax, zmin=-1,
	zmax=1, xname="Time, ms", yname="z, $\\mu$m", zname="Spin projection")
pr = HeightmapPlot(pr)
pr.save('test.pdf')
