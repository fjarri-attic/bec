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

env = Environment(gpu=True)
constants = Constants(Model(N=150000), double_precision=False)

cloud = GPEGroundState(env, constants).createCloud()
evolution = TwoComponentEvolution(env, constants)
pulse = Pulse(env, constants)

pulse.halfPi(cloud)
t_to_equality = evolution.run(cloud, time=0.05,
	callbacks=[ParticleNumberCondition(env, constants, ratio=0.15, verbose=True)])
print t_to_equality
pulse.apply(cloud, theta=1.5*math.pi, phi=0)

evolution.run(cloud, time=0.299, callbacks=[sp], callback_dt=0.01)
sp = SurfaceProjectionCollector(env, constants, do_pulse=False)

env.synchronize()

times, a_xy, a_yz, b_xy, b_yz = sp.getData()

#times, picture = a.getData()
#pr = HeightmapData("test", picture, xmin=0, xmax=100,
#	ymin=-constants.zmax, ymax=constants.zmax, zmin=-1,
#	zmax=1, xname="Time, ms", yname="z, $\\mu$m", zname="Spin projection")
#pr = HeightmapPlot(pr)
#pr.save('test.pdf')

times = [str(int(x * 1000 + 0.5)) for x in times]

for name, dataset in (('testa.pdf', a_yz), ('testb.pdf', b_yz)):
	hms = []
	for t, hm in zip(times, dataset):
		hms.append(HeightmapData(t, hm.transpose(),
			xmin=-constants.zmax, xmax=constants.zmax,
			ymin=-constants.ymax, ymax=constants.ymax,
			zmin=0, zmax=400))

	EvolutionPlot(hms, shape=(6, 5)).save(name)
