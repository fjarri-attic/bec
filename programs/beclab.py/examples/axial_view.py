import numpy
import time
import math

from beclab import *

def testAxial(gpu, ideal_pulses):
	# preparation
	env = Environment(gpu=gpu)
	constants = Constants(Model(N=150000, detuning=-41),
		double_precision=False if gpu else True)

	gs = GPEGroundState(env, constants)
	evolution = SplitStepEvolution(env, constants)
	pulse = Pulse(env, constants)
	a = AxialProjectionCollector(env, constants, ideal_pulse=ideal_pulses, pulse=pulse)

	# experiment
	cloud = gs.createCloud()

	if ideal_pulses:
		pulse.applyInstantaneous(cloud, theta=0.5 * math.pi)
	else:
		pulse.apply(cloud, theta=0.5 * math.pi)

	t1 = time.time()
	evolution.run(cloud, time=0.1, callbacks=[a], callback_dt=0.005)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	times, picture = a.getData()

	return HeightmapPlot(
		HeightmapData("test", picture,
			xmin=0, xmax=100,
			ymin=-constants.zmax * constants.l_rho * 1e6,
			ymax=constants.zmax * constants.l_rho * 1e6,
			zmin=-1, zmax=1,
			xname="Time, ms", yname="z, $\\mu$m", zname="Spin projection")
	)

for gpu, ideal_pulses in ((False, True), (False, False), (True, True), (True, False)):
	suffix = ("gpu" if gpu else "cpu") + "_" + ("ideal" if ideal_pulses else "nonideal") + "_pulses"
	testAxial(gpu=gpu, ideal_pulses=ideal_pulses).save("axial_" + suffix + ".pdf")
