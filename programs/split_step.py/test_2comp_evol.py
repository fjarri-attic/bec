import numpy
import time
import math

from globals import Environment
from model import Model
from constants import Constants
from evolution import TwoComponentBEC
import typenames

from collectors import *

from datahelpers import XYData, HeightmapData, XYPlot, HeightmapPlot, EvolutionPlot

m = Model()
constants = Constants(m)
env = Environment(True, typenames.single_precision, constants)
bec = TwoComponentBEC(env)

e = EqualParticleNumberCondition(env)
t_to_equality = bec.runEvolution(0.2, [e])
env.synchronize()

print t_to_equality

a = AxialProjectionCollector(env, do_pulse=False)
p = ParticleNumberCollector(env, verbose=True, do_pulse=False)
sc = SliceCollector(env)
sp = SurfaceProjectionCollector(env)

bec._pulse.apply(bec._a, bec._b, 3.0 * math.pi / 2.0, 0)
bec.runEvolution(0.199, [sp], callback_dt=0.01)
env.synchronize()

times, a_xy, a_yz, b_xy, b_yz = sp.getData()

#times, picture = a.getData()
#pr = HeightmapData("test", picture, xmin=0, xmax=100,
#	ymin=-env.constants.zmax, ymax=env.constants.zmax, zmin=-1,
#	zmax=1, xname="Time, ms", yname="z, $\\mu$m", zname="Spin projection")
#pr = HeightmapPlot(pr)
#pr.save('test.pdf')

times = [str(int(x * 1000 + 0.5)) for x in times]

for name, dataset in (('testa.pdf', a_yz), ('testb.pdf', b_yz)):
	hms = []
	for t, hm in zip(times, dataset):
		hms.append(HeightmapData(t, hm.transpose(),
			xmin=-env.constants.zmax, xmax=env.constants.zmax,
			ymin=-env.constants.ymax, ymax=env.constants.ymax,
			zmin=0, zmax=400))

	EvolutionPlot(hms, shape=(5, 4)).save(name)
