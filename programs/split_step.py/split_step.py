import matplotlib.pyplot as plt
import numpy
import time

from globals import Environment
from model import Model
from constants import Constants
from evolution import TwoComponentBEC
import typenames
from colormap import blue_white_red

from collectors import *

from datahelpers import XYData, HeightmapData


for gpu in (False,):
	m = Model()
	constants = Constants(m)
	env = Environment(gpu, typenames.double_precision, constants)

	bec = TwoComponentBEC(env)

	vis = VisibilityCollector(env, verbose=True)
	#s = SurfaceProjectionCollector(env)
	#a = AxialProjectionCollector(env)
	#p = ParticleNumberCollector(env, verbose=True)

	t1 = time.time()
	bec.runEvolution(0.6, [vis], callback_dt=0.01)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	times, visibility = vis.getData()
	xy = XYData("30k atoms, 8x8x64, 8 ensembles", times, visibility, ymin=0, ymax=1, xname="Time, s",
		yname="Visibility", source="8x8x64, 8 ensembles, CPU, double precision")
	xy.save("test1.yaml")

	#times, Na, Nb, N = p.getData()
	#xy = XYData("30k atoms, no noise", times, N, ymin=0, ymax=30000, xname="Time, s",
	#	yname="N", source="16x16x128, GPU, double precision")
	#xy.save("test1.yaml")

	#times, a_xy, a_yz, b_xy, b_yz = s.getData()
	#times, picture = a.getData()
