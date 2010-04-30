import numpy
import time
import math

from globals import Environment
from model import Model
from constants import Constants
from evolution import TwoComponentBEC
from ground_state import GPEGroundState
from collectors import SurfaceProjectionCollector
import typenames

from collectors import *

from datahelpers import XYData, HeightmapData, XYPlot, HeightmapPlot

m = Model()
constants = Constants(m)
env = Environment(False, typenames.single_precision, constants)

gs = GPEGroundState(env)

a, b = gs.create(two_component=True)
sp = SliceCollector(env)

sp(0, a, b)
times, a_xy, a_yz, b_xy, b_yz = sp.getData()


a_data = HeightmapData("1 component", a_yz[0].transpose(), xmin=-env.constants.zmax, xmax=env.constants.zmax,
	xname="Z, $\\mu$m", yname="Y, $\\mu$m",
	ymin=-env.constants.ymax, ymax=env.constants.ymax, zmin=0)
HeightmapPlot(a_data).save('test_a.pdf')

b_data = HeightmapData("2 component", b_yz[0].transpose(), xmin=-env.constants.zmax, xmax=env.constants.zmax,
	xname="Z, $\\mu$m", yname="Y, $\\mu$m",
	ymin=-env.constants.ymax, ymax=env.constants.ymax, zmin=0)
HeightmapPlot(b_data).save('test_b.pdf')
