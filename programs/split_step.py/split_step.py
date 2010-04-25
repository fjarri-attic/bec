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


for gpu in (False,):
	m = Model()
	constants = Constants(m)
	env = Environment(gpu, typenames.double_precision, constants)

	bec = TwoComponentBEC(env)

	#vis = VisibilityCollector(env)
	#s = SurfaceProjectionCollector(env)
	#a = AxialProjectionCollector(env)
	#p = ParticleNumberCollector(env)
	b = BlochSphereCollector(env, amp_range=(math.pi * (1.0/2 - 1.0/6), math.pi * (1.0/2 + 1.0/6)))

	t1 = time.time()
	bec.runEvolution(0.02, [b], callback_dt=0.005)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	times, snapshots = b.getData()
	for t, snapshot in zip(times, snapshots):
		pr = HeightmapData("BS projection test", snapshot, xmin=0, xmax=2 * math.pi,
			ymin=math.pi * (1.0/2 - 1.0/6), ymax=math.pi * (1.0/2 + 1.0/6),
			zmin=0, xname="Phase", yname="Amplitude", zname="Density")
		pr = HeightmapPlot(pr)
		pr.save('test' + str(int(t * 1000 + 0.5)).zfill(3) + '.png')

	#times, visibility = vis.getData()
	#xy = XYData("30k atoms, 16x16x128, 4 ensembles, new noise", times, visibility, ymin=0, ymax=1, xname="Time, s",
	#	yname="Visibility")
	#xy.save("test.yaml")

	#times, Na, Nb, N = p.getData()
	#xy = XYData("30k atoms, no noise", times, N, ymin=0, ymax=30000, xname="Time, s",
	#	yname="N", source="16x16x128, GPU, double precision")
	#xy.save("test1.yaml")

	#times, picture = a.getData()
	#ax = HeightmapData("test", picture, xmin=0, xmax=times[-1], xname="Time,ms", ymin=-env.constants.zmax,
	#	ymax=env.constants.zmax, yname="z, $mu$m", zmin=-1, zmax=1)
	#ax = HeightmapPlot(ax)
	#ax.save('test.pdf')

	#times, Na, Nb, N = p.getData()
	#times, v = vis.getData()
	#Diff = (Na - Nb) / env.constants.N
	#xy1 = XYData("Fringes", times, Diff, ymin=-1, ymax=1, xname="Time, s", yname="(Na - Nb) / N")
	#xy2 = XYData("Visibility", times, v, ymin=0, ymax=1, xname="Time, s", yname="Visibility")
	#plt = XYPlot([xy1, xy2])
	#plt.save('test.pdf')
