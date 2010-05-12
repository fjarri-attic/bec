import numpy
import time
import math

from globals import Environment
from model import Model
from constants import Constants
from evolution import TwoComponentEvolution
from ground_state import GPEGroundState

from collectors import *

from datahelpers import XYData, HeightmapData, XYPlot, HeightmapPlot

# preparation
env = Environment(gpu=False)
constants = Constants(Model(N=150000), double_precision=False)
gs = GPEGroundState(env, constants)
sp = SliceCollector(env, constants, do_pulse=False)

# experiment
cloud = gs.createCloud(two_component=True)
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
