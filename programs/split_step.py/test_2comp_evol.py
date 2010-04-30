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

e = EqualParticleNumberCondition(env)
t_to_equality = bec.runEvolution(0.05, [e])
env.synchronize()

print t_to_equality

a = AxialProjectionCollector(env, do_pulse=False)
p = ParticleNumberCollector(env, verbose=True, do_pulse=False)
sc = SliceCollector(env)
sp = SurfaceProjectionCollector(env)

bec._pulse.halfPi(bec._a, bec._b)
bec.runEvolution(0.1, [a, p], callback_dt=0.001)
env.synchronize()

sp(0, bec._a, bec._b)
times, a_xy, a_yz, b_xy, b_yz = sp.getData()
a_data = HeightmapData("1 component", a_yz[0].transpose(), xmin=-env.constants.zmax, xmax=env.constants.zmax,
	xname="Z, $\\mu$m", yname="Y, $\\mu$m",
	ymin=-env.constants.ymax, ymax=env.constants.ymax, zmin=0)
HeightmapPlot(a_data).save('test_a.pdf')

b_data = HeightmapData("2 component", b_yz[0].transpose(), xmin=-env.constants.zmax, xmax=env.constants.zmax,
	xname="Z, $\\mu$m", yname="Y, $\\mu$m",
	ymin=-env.constants.ymax, ymax=env.constants.ymax, zmin=0)
HeightmapPlot(b_data).save('test_b.pdf')

times, picture = a.getData()
pr = HeightmapData("test", picture, xmin=0, xmax=100,
	ymin=-env.constants.zmax, ymax=env.constants.zmax, zmin=-1,
	zmax=1, xname="Time, ms", yname="z, $\\mu$m", zname="Spin projection")
pr = HeightmapPlot(pr)
pr.save('test.pdf')
