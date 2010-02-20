try:
	import pycudafft
except:
	pass

import numpy

class NumpyPlan:

	def __init__(self, x, y, z, dtype):
		self._x = x
		self._y = y
		self._z = z
		self._dtype = dtype

	def execute(self, data_in, inverse=False, data_out=None, batch=1):
		res = numpy.empty(data_in.shape, dtype=self._dtype)

		func = numpy.fft.ifftn if inverse else numpy.fft.fftn

		for i in xrange(batch):
			start = i * self._z
			stop = (i + 1) * self._z
			res[start:stop,:,:] = func(data_in[start:stop,:,:])

		if data_out is None:
			data_in[:,:,:] = res
		else:
			data_out[:,:,:] = res

def createPlan(gpu, x, y, z, batch, precision):
	if gpu:
		return pycudafft.FFTPlan(x, y=y, z=z, precision=precision.fft_precision, normalize=True)
	else:
		return NumpyPlan(x, y, z, dtype=precision.dtype)
