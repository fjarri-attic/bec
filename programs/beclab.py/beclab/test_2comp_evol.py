import numpy
import time
import math

from globals import Environment
from model import Model
from constants import Constants, COMP_1_minus1, COMP_2_1
from evolution import TwoComponentEvolution
from ground_state import GPEGroundState

from collectors import *

from datahelpers import XYData, HeightmapData, XYPlot, HeightmapPlot, EvolutionPlot

env = Environment(gpu=False)

aa = [
	(100.4, 95.0, 97.66),
#	(99.71, 95.66, 97.66),
#	(99.02, 96.32, 97.66),
#	(98.35, 96.98, 97.66)
]

for a11, a22, a12 in aa:
	constants = Constants(Model(N=150000, a11=a11, a12=a12, a22=a22), double_precision=True)
	fc = (constants.muTF(comp=COMP_1_minus1, N=constants.N/2) -
		constants.muTF(comp=COMP_2_1, N=constants.N / 2)) / constants.t_rho / (2.0 * math.pi)
	print fc
	constants = Constants(Model(N=150000, a11=a11, a12=a12, a22=a22, detuning=-fc,
		dt_evo=1e-5), double_precision=True)

	cloud = GPEGroundState(env, constants).createCloud()
	evolution = TwoComponentEvolution(env, constants)
	pulse = Pulse(env, constants)

	pulse.halfPi(cloud)

	t_to_equality = evolution.run(cloud, time=0.03,
		callbacks=[ParticleNumberCondition(env, constants, ratio=0.5)])
	print t_to_equality
	pulse.apply(cloud, theta=0.5 * math.pi, phi=0)
	cloud.b._fillWithZeros()
	sp = SurfaceProjectionCollector(env, constants, do_pulse=False)
	#a = AxialViewCollector(env, constants, do_pulse=True)
	#v = VisibilityCollector(env, constants)
	evolution.run(cloud, time=0.4, callbacks=[sp], callback_dt=0.02)

	env.synchronize()

	#times, vis = v.getData()
	#vis = XYData("test", times, vis, ymin=0, ymax=1, xname="Time, s", yname="Visibility")
	#vis = XYPlot([vis])
	#vis.save('test.pdf')

	#times, picture = a.getData()
	#pr = HeightmapData("test", picture, xmin=0, xmax=100,
	#	ymin=-constants.zmax, ymax=constants.zmax, zmin=-1,
	#	zmax=1, xname="Time, ms", yname="z, $\\mu$m", zname="Spin projection")
	#pr = HeightmapPlot(pr)
	#pr.save('test.pdf')

	times, a_xy, a_yz, b_xy, b_yz = sp.getData()
	times = [str(int(x * 1000 + 0.5)) for x in times]

	name = '150k_' + str(a11) + '_' + str(a22) + '_' + str(a12) + '_' + str(-fc) + '_pi2_'
	for name, dataset, zmax in ((name + 'a.pdf', a_yz, 100), (name + 'b.pdf', b_yz, 600)):
		hms = []
		for t, hm in zip(times, dataset):
			hms.append(HeightmapData(t, hm.transpose(),
				xmin=-constants.zmax, xmax=constants.zmax,
				ymin=-constants.ymax, ymax=constants.ymax,
				zmin=0, zmax=zmax))

		EvolutionPlot(hms, shape=(5, 4)).save(name)
