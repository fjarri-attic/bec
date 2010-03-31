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


for gpu in (False,):
	m = Model()
	constants = Constants(m)
	env = Environment(gpu, typenames.single_precision, constants)

	print str(env)

	bec = TwoComponentBEC(env)

	vis = VisibilityCollector(env)
	s = SurfaceProjectionCollector(env)
	a = AxialProjectionCollector(env)

	t1 = time.time()
	bec.runEvolution(0.01, [vis, s, a], callback_dt=1)
	env.synchronize()
	t2 = time.time()
	print "Time spent: " + str(t2 - t1) + " s"

	times, visibility = vis.getData()
	times, a_xy, a_yz, b_xy, b_yz = s.getData()
	times, picture = a.getData()

	for i, e in enumerate([a_xy[0], a_yz[0], b_xy[0], b_yz[0]]):
		z = numpy.array(e)
		z = z.transpose()

		plt.figure()
	#	im = plt.imshow(z, interpolation='bilinear', origin='lower',
	#		aspect='auto', extent=(0, times[-1], -constants.zmax, constants.zmax), cmap=blue_white_red)
		im = plt.imshow(z, interpolation='nearest', origin='lower',
			aspect='auto', extent=(-constants.ymax, constants.ymax, -constants.zmax, constants.zmax), cmap=blue_white_red)

		CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)
		plt.xlabel('Time, ms')
		plt.ylabel('z, $\mu$m')
		plt.savefig('pr' + str(i) + '.pdf')
