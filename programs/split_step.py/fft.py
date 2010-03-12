try:
	import pycudafft
except:
	pass

import numpy

class NumpyPlan:

	def __init__(self, x, y, z):
		self._x = x
		self._y = y
		self._z = z

	def execute(self, data_in, data_out=None, inverse=False, batch=1):
		res = numpy.empty(data_in.shape, dtype=data_in.dtype)

		if inverse:
			func = numpy.fft.ifftn
		else:
			func = numpy.fft.fftn

		for i in xrange(batch):
			start = i * self._z
			stop = (i + 1) * self._z
			res[start:stop,:,:] = func(data_in[start:stop,:,:])

		if data_out is None:
			data_in[:,:,:] = res
		else:
			data_out[:,:,:] = res

def createPlan(gpu, x, y, z, precision):
	if gpu:
		return pycudafft.FFTPlan((x, y, z), dtype=precision.complex.dtype, normalize=True)
	else:
		return NumpyPlan(x, y, z)
