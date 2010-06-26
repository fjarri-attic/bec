import numpy
import time
import math

from globals import Environment
from model import Model
from constants import Constants, COMP_1_minus1, COMP_2_1
from evolution import TwoComponentEvolution
from ground_state import GPEGroundState
from state import ParticleStatistics

from collectors import *

from datahelpers import XYData, HeightmapData, XYPlot, HeightmapPlot

# preparation
env = Environment(gpu=False)
constants = Constants(Model(N=150000, nvx=32, nvy=32), double_precision=False)
gs = GPEGroundState(env, constants)
sp = SliceCollector(env, constants, do_pulse=False)

print "TF:"
print constants.muTF(comp=COMP_1_minus1), constants.muTF(comp=COMP_2_1), (constants.muTF(comp=COMP_1_minus1) - constants.muTF(comp=COMP_2_1)) / constants.t_rho / (2 * math.pi)
exit()
# experiment
state1 = gs.createState(comp=COMP_1_minus1)
state2 = gs.createState(comp=COMP_2_1)

stats = ParticleStatistics(env, constants)

mu1 = stats.countMu(state1)
mu2 = stats.countMu(state2)
fc = (mu1 - mu2) / constants.t_rho / (2 * math.pi)

print "Separate GPE:"
print mu1, mu2, fc

cloud = gs.createCloud(two_component=True)

#mu1 = stats._countStateTwoComponent(state1, state2, 2, constants.N)
#mu2 = stats._countStateTwoComponent(state2, state1, 2, constants.N)
#fc = (mu2 - mu1) / constants.t_rho / (2 * math.pi)
#print "Coupled GPE:"
#print mu1, mu2, fc

sp(0, cloud)

# render
times, a_xy, a_yz, b_xy, b_yz = sp.getData()

a_data = HeightmapData("1 component", a_yz[0].transpose(), xmin=-constants.zmax, xmax=constants.zmax,
	xname="Z, $\\mu$m", yname="Y, $\\mu$m",
	ymin=-constants.ymax, ymax=constants.ymax, zmin=0)
HeightmapPlot(a_data).save('test_a.pdf')

b_data = HeightmapData("2 component", b_yz[0].transpose(), xmin=-constants.zmax, xmax=constants.zmax,
	xname="Z, $\\mu$m", yname="Y, $\\mu$m",
	ymin=-constants.ymax, ymax=constants.ymax, zmin=0)
HeightmapPlot(b_data).save('test_b.pdf')
