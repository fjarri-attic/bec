"""
Module, containing class with calculation constants
"""

import math

from globals import *

class Constants:
	"""Calculation constants, in natural units"""

	def __init__(self, model):
		w_rho = 2.0 * math.pi * model.fx # radial oscillator frequency
		l_rho = math.sqrt(model.hbar / (model.m * w_rho)) # natural length
		self.l_rho = l_rho
		self.lambda_ = model.fx / model.fz

		self.t_rho = 1.0 / w_rho # natural time unit

		self.nvx = model.nvx
		self.nvy = model.nvy
		self.nvz = model.nvz
		self.cells = self.nvx * self.nvy * self.nvz
		self.shape = (self.nvz, self.nvy, self.nvx)

		self.V1 = model.V1
		self.V2 = model.V2
		self.V = (self.V1 + self.V2) / 2.0

		self.detuning = 2 * math.pi * model.detuning / w_rho

		self.l111 = model.gamma111 / (pow(l_rho, 6) * w_rho)
		self.l12 = model.gamma12 / (pow(l_rho, 3) * w_rho)
		self.l22 = model.gamma22 / (pow(l_rho, 3) * w_rho)

		self.g11 = 4 * math.pi * model.a11 * model.a0 / l_rho
		self.g12 = 4 * math.pi * model.a12 * model.a0 / l_rho
		self.g22 = 4 * math.pi * model.a22 * model.a0 / l_rho

		self.N = model.N

		# TF-approximated chemical potentials
		self.mu = (15.0 * self.N * self.g11 / (16.0 * math.pi * self.lambda_ * math.sqrt(2.0))) ** 0.4
		self.mu2 = (15.0 * self.N * self.g22 / (16.0 * math.pi * self.lambda_ * math.sqrt(2.0))) ** 0.4

		self.xmax = model.border * math.sqrt(2.0 * self.mu)
		self.ymax = self.xmax
		self.zmax = self.xmax * self.lambda_

		# space step
		self.dx = 2.0 * self.xmax / (self.nvx - 1)
		self.dy = 2.0 * self.ymax / (self.nvy - 1)
		self.dz = 2.0 * self.zmax / (self.nvz - 1)
		self.dV = self.dx * self.dy * self.dz

		self.nvx_pow = log2(self.nvx)
		self.nvy_pow = log2(self.nvy)
		self.nvz_pow = log2(self.nvz)

		# k step
		self.dkx = math.pi / self.xmax
		self.dky = math.pi / self.ymax
		self.dkz = math.pi / self.zmax

		self.itmax = model.itmax
		self.dt_steady = model.dt_steady / self.t_rho
		self.t_equilib = model.t_equilib / self.t_rho
		self.dt_evo = model.dt_evo / self.t_rho
		self.ensembles = model.ensembles
		self.ens_shape = (self.ensembles * self.nvz, self.nvy, self.nvx)
