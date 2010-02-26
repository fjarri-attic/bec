import numpy

try:
	from pycudafft import SINGLE_PRECISION, DOUBLE_PRECISION
except:
	SINGLE_PRECISION = None
	DOUBLE_PRECISION = None

class _Type:
	def __init__(self, name, dtype, ctype):
		self.name = name
		self.dtype = dtype
		self.nbytes = dtype().nbytes
		self.ctype = ctype
		self.ctr = 'make_' + name

class _Precision:
	def __init__(self, scalar, complex, fft_precision):
		self.scalar = scalar
		self.complex = complex
		self.fft_precision = fft_precision

_single_float = _Type('float', numpy.float32, "f")
_double_float = _Type('double', numpy.float64, "d")

_single_complex = _Type('float2', numpy.complex64, "d")
_double_complex = _Type('double2', numpy.complex128, None)

single_precision = _Precision(_single_float, _single_complex, SINGLE_PRECISION)
double_precision = _Precision(_double_float, _double_complex, DOUBLE_PRECISION)
