import numpy

class _Type:
	def __init__(self, name, dtype):
		self.name = name
		self.dtype = dtype
		self.nbytes = dtype().nbytes
		self.ctr = '(' + name + ')'
		self.cast = numpy.cast[dtype]

class _Precision:
	def __init__(self, scalar, complex):
		self.scalar = scalar
		self.complex = complex

_single_float = _Type('float', numpy.float32)
_double_float = _Type('double', numpy.float64)

_single_complex = _Type('float2', numpy.complex64)
_double_complex = _Type('double2', numpy.complex128)

single_precision = _Precision(_single_float, _single_complex)
double_precision = _Precision(_double_float, _double_complex)
