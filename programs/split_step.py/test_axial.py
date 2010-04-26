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

a = AxialProjectionCollector(env)
t1 = time.time()
bec.runEvolution(0.099, [a], callback_dt=0.001)
env.synchronize()
t2 = time.time()
print "Time spent: " + str(t2 - t1) + " s"

times, picture = a.getData()

pr = HeightmapData("test", picture, xmin=0, xmax=100,
	ymin=-env.constants.zmax, ymax=env.constants.zmax, zmin=-1,
	zmax=1, xname="Time, ms", yname="z, $\\mu$m", zname="Spin projection")
pr = HeightmapPlot(pr)
pr.save('test.pdf')
