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
env = Environment(True, typenames.single_precision, constants)
bec = TwoComponentBEC(env)

a = VisibilityCollector(env, verbose=True)
t1 = time.time()
bec.runEvolution(0.099, [a], callback_dt=0.01)
env.synchronize()
t2 = time.time()
print "Time spent: " + str(t2 - t1) + " s"

times, vis = a.getData()

vis = XYData("GPU, noise with equilibration", times, vis, ymin=0, ymax=1,
	xname="Time, ms", yname="Visibility")
vis.save('test.yaml')
vis = XYPlot([vis])
vis.save('test.pdf')
