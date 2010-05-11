import numpy
import time
import math

from globals import Environment
from model import Model
from constants import Constants
from evolution import TwoComponentEvolution, Pulse
from ground_state import GPEGroundState

from collectors import *

from datahelpers import XYData, HeightmapData, XYPlot, HeightmapPlot

constants = Constants(Model(), double_precision=False)
env = Environment(gpu=True)
evolution = TwoComponentEvolution(env, constants)
a = VisibilityCollector(env, constants, verbose=True)
b = ParticleNumberCollector(env, constants, verbose=True)

gs = GPEGroundState(env, constants)
pulse = Pulse(env, constants)

cloud = gs.create()
pulse.halfPi(cloud)
t1 = time.time()
evolution.run(cloud, 0.0399, callbacks=[a, b], callback_dt=0.01)
env.synchronize()
t2 = time.time()
print "Time spent: " + str(t2 - t1) + " s"

times, vis = a.getData()

vis = XYData("GPU, noise with equilibration", times, vis, ymin=0, ymax=1,
	xname="Time, ms", yname="Visibility")
vis.save('test.yaml')
vis = XYPlot([vis])
vis.save('test.pdf')
