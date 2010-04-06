import matplotlib.pyplot as plt
from colormap import blue_white_red


class HeightmapPlot:

	def __init__(self, heightmap_data):
		self.data = heightmap_data

		self.fig = plt.figure()

		subplot = self.fig.add_subplot(111, xlabel=self.data.x_name, ylabel=self.data.y_name)
		im = subplot.imshow(self.data.heightmap, interpolation='bilinear', origin='lower',
			aspect='auto', extent=(self.data.x_min, self.data.x_max,
			self.data.y_min, self.data.y_max), cmap=blue_white_red,
			vmin=self.data.z_min, vmax=self.data.z_max)
		self.fig.colorbar(im, orientation='horizontal', shrink=0.8)

	def save(self, filename):
		self.fig.savefig(filename)


class XYPlot:

	def __init__(self, xy_data_list):
		self.data_list = xy_data_list

		# check that data contains the same values
		x_name = self.data_list[0].x_name
		y_name = self.data_list[0].y_name
		for data in self.data_list:
			assert data.x_name == x_name
			assert data.y_name == y_name

		# find x limits
		x_min = self.data_list[0].x_array[0]
		x_max = self.data_list[0].x_array[-1]
		for data in self.data_list:
			if data.x_array[0] < x_min:
				x_min = data.x_array[0]
			if data.x_array[-1] > x_max:
				x_max = data.x_array[-1]

		# find y limits
		y_min = None
		y_max = None
		for data in self.data_list:
			if data.y_min is not None and (y_min is None or data.y_min < y_min):
				y_min = data.y_min
			if data.y_max is not None and (y_max is None or data.y_max < y_max):
				y_max = data.y_max

		# plot data
		self.fig = plt.figure()

		subplot = self.fig.add_subplot(111,
			xlabel=self.data_list[0].x_name,
			ylabel=self.data_list[0].y_name)

		for data in self.data_list:
			subplot.plot(data.x_array, data.y_array, label=data.name)

		subplot.set_xlim(xmin=x_min, xmax=x_max)
		subplot.set_ylim(ymin=y_min, ymax=y_max)

		subplot.legend(loc=3, prop={'size': 'x-small'})

		subplot.grid(True)

	def save(self, filename):
		self.fig.savefig(filename)

