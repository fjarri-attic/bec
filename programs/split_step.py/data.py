import yaml
import numpy


class Data:

	def __init__(self, name, description, source):
		self.name = name
		self.description = description
		self.source = source
		self.format = _FORMATS[self.__class__]

	def dump(self, filename):
		string = yaml.dump({'name': self.name, 'format': self.format, 'data': self._getData(),
			'description': self.description, 'source': self.source},
			default_flow_style=False)
		open(filename, "w").write(string)


class XYData(Data):

	def __init__(self, name, x_array, y_array, description=None,
		source=None, y_min=None, y_max=None, x_name=None, y_name=None):

		Data.__init__(self, name, description, source)

		assert isinstance(x_array, numpy.ndarray)
		assert isinstance(y_array, numpy.ndarray)
		assert x_array.size == y_array.size

		self.x_array = x_array
		self.y_array = y_array
		self.y_min = float(y_min)
		self.y_max = float(y_max)
		self.x_name = x_name
		self.y_name = y_name

	def _getData(self):
		return {
			'x_array': repr(self.x_array.tolist()),
			'y_array': repr(self.y_array.tolist()),
			'y_min': self.y_min,
			'y_max': self.y_max,
			'x_name': self.x_name,
			'y_name': self.y_name
		}

	@classmethod
	def load(cls, contents):
		data = contents['data']
		x_array = numpy.array(eval(data.pop('x_array')))
		y_array = numpy.array(eval(data.pop('y_array')))
		return cls(contents['name'], x_array, y_array, description=contents['description'],
			source=contents['source'], **data)


class HeightmapData(Data):

	def __init__(self, name, heightmap, x_min=None, x_max=None, y_min=None, y_max=None, description=None,
		source=None, z_min=None, z_max=None, x_name=None, y_name=None, z_name=None):

		Data.__init__(self, name, description, source)

		assert isinstance(heightmap, numpy.ndarray)

		self.heightmap = heightmap
		self.x_min = float(x_min)
		self.x_max = float(x_max)
		self.y_min = float(y_min)
		self.y_max = float(y_max)
		self.z_min = float(z_min)
		self.z_max = float(z_max)
		self.x_name = x_name
		self.y_name = y_name
		self.z_name = z_name

	def _getData(self):
		return {
			'heightmap': repr(self.heightmap.tolist()),
			'x_min': self.x_min,
			'x_max': self.x_max,
			'y_min': self.y_min,
			'y_max': self.y_max,
			'z_min': self.z_min,
			'z_max': self.z_max,
			'x_name': self.x_name,
			'y_name': self.y_name,
			'z_name': self.z_name
		}

	@classmethod
	def load(cls, contents):
		data = contents['data']
		heightmap = numpy.array(eval(data.pop('heightmap')))
		return cls(contents['name'], heightmap, description=contents['description'],
			source=contents['source'], **data)


_CLASSES = {
	'xy': XYData,
	'heightmap': HeightmapData
}

_FORMATS = dict((v, k) for k, v in _CLASSES.iteritems())


def load(filename):
	string = open(filename).read()
	contents = yaml.load(string)

	format = contents['format']
	if format not in _CLASSES:
		raise Exception("Unknown data format: " + str(format))

	cls = _CLASSES[format]
	return cls.load(contents)
