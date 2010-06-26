import numpy
import time
import math

from globals import Environment
from model import Model
from constants import Constants, COMP_1_minus1, COMP_2_1
from evolution import TwoComponentEvolution, Pulse
from ground_state import GPEGroundState

from collectors import *

from datahelpers import XYData, HeightmapData, XYPlot, HeightmapPlot

constants = Constants(Model(N=30000, nvx=16, nvy=16, nvz=128), double_precision=True)
env = Environment(gpu=False)
evolution = TwoComponentEvolution(env, constants)
a = VisibilityCollector(env, constants, verbose=True)
b = ParticleNumberCollector(env, constants, verbose=True)
sp = SurfaceProjectionCollector(env, constants)

gs = GPEGroundState(env, constants)
pulse = Pulse(env, constants)

cloud = gs.createCloud()

#pulse.halfPi(cloud)
pulse.applyNonIdeal(cloud, math.pi * 0.5)

t1 = time.time()
evolution.run(cloud, 0.1, callbacks=[a, b, sp], callback_dt=0.01)
env.synchronize()
t2 = time.time()
print "Time spent: " + str(t2 - t1) + " s"

"""
times, vis = a.getData()
vis = XYData("test", times, vis, ymin=0, ymax=1, xname="Time, s", yname="Visibility")
vis = XYPlot([vis])
vis.save('test.pdf')
"""


"""
times, a_xy, a_yz, b_xy, b_yz = sp.getData()
HeightmapPlot(HeightmapData("test", a_yz[0],
	xmin=-constants.zmax, xmax=constants.zmax,
	ymin=-constants.ymax, ymax=constants.ymax
)).save('testa.pdf')
HeightmapPlot(HeightmapData("test", b_yz[0],
	xmin=-constants.zmax, xmax=constants.zmax,
	ymin=-constants.ymax, ymax=constants.ymax
)).save('testb.pdf')
"""