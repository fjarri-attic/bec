import numpy
import time
import math

from globals import Environment
from model import Model
from constants import Constants, COMP_1_minus1, COMP_2_1
from evolution import TwoComponentEvolution
from ground_state import GPEGroundState

from collectors import *

from datahelpers import XYData, HeightmapData, XYPlot, HeightmapPlot

# preparation
env = Environment(gpu=False)

constants = Constants(Model(N=150000, detuning=-41), double_precision=True)

gs = GPEGroundState(env, constants)
evolution = TwoComponentEvolution(env, constants)
pulse = Pulse(env, constants)
a = AxialProjectionCollector(env, constants)

# experiment
cloud = gs.createCloud()
#pulse.apply(cloud, theta=0.5*math.pi, phi=0)
pulse.applyNonIdeal(cloud, math.pi * 0.5)
t1 = time.time()
evolution.run(cloud, time=0.1, callbacks=[a], callback_dt=0.005)
env.synchronize()
t2 = time.time()
print "Time spent: " + str(t2 - t1) + " s"

times, picture = a.getData()

pr = HeightmapData("test", picture, xmin=0, xmax=600,
	ymin=-constants.zmax, ymax=constants.zmax, zmin=-1,
	zmax=1, xname="Time, ms", yname="z, $\\mu$m", zname="Spin projection")
pr = HeightmapPlot(pr)
pr.save('test.pdf')
