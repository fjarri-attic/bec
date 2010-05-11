"""
Split-step calculation parameters
"""

class Model:
	"""
	Model parameters;
	in SI units, unless explicitly specified otherwise
	"""

	hbar = 1.054571628e-34 # Planck constant
	a0 = 5.2917720859e-11 # Bohr radius

	N = 150000 # number of particles
	m = 1.443160648e-25 # mass of one particle (rubidium-87)

	# scattering lengths, in Bohr radii

	# from AS presentation
	#a11 = 100.44;
	#a22 = 95.47;
	#a12 = 98.09;

	a11 = 100.4
	a22 = 95.0
	a12 = 97.66

	# Trap frequencies
	fx = 97.6
	fy = 97.6
	fz = 11.96

	# detuning frequency
	detuning = -41

	# loss terms
	gamma111 = 5.4e-42
	gamma12 = 0.78e-19
	gamma22 = 1.194e-19
	#gamma111 = 0
	#gamma12 = 0
	#gamma22 = 0

	# spatial lattice size
	nvx = 16
	nvy = 16
	nvz = 128

	# number of iterations for integration in mid step
	itmax = 3

	dt_steady = 0.00002 # time step for steady state calculation
	t_equilib = 0 # equilibration time
	dt_evo = 0.00004 # time step for evolution

	ensembles = 4 # number of ensembles

	border = 1.2 # defines, how big is calculation area as compared to cloud size

	def __init__(self, **kwds):
		for kwd in kwds:
			self.__dict__[kwd] = kwds[kwd]
