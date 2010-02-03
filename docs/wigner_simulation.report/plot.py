import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

cmap_file = open("TemperatureMap.tsv")
cmap_list = []
for line in cmap_file:
	elems = [float(x) / 256 for x in line.split("\t")]
	cmap_list.append(elems)
cmap = ListedColormap(cmap_list, name="BlueWhiteRed")

def load_xy_data(file_name):
	f = open(file_name)
	x = []
	y = []
	for line in f:
		elems = line.split(' ')
		x.append(float(elems[0]))
		y.append(float(elems[1]))
	f.close()
	return x, y

def load_heightmap_data(file_name):
	f = open(file_name)
	times = []
	arr = []
	for line in f:
		elems = line.split(' ')
		times.append(float(elems[0]))
		arr.append([float(x) for x in elems[1:]])
	f.close()
	return times, arr

# plot visibility

no_noise_x, no_noise_y = load_xy_data('visibility.txt')
noise_x, noise_y = load_xy_data('visibility_noise.txt')

plt.plot(no_noise_x, no_noise_y, label='Simulation, no quantum noise')
plt.plot(noise_x, noise_y, label='Simulation, with quantum noise')
plt.legend()
plt.xlabel('Time, ms')
plt.ylabel('Visibility')
plt.axis([0, 600, 0, 1])
plt.grid(True)
plt.savefig('visibility.pdf')

# plot evolution

times, z = load_heightmap_data('axial_no_noise.txt')
z = np.array(z)
z = z.transpose()
plt.figure()
im = plt.imshow(z, interpolation='bilinear', origin='lower',
	aspect='auto', extent=(0, times[-1], -45, 45), cmap=cmap)
CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)
plt.xlabel('Time, ms')
plt.ylabel('z, $\mu$m')
plt.savefig('axial_view.pdf')

times, z = load_heightmap_data('axial_pi_pulse.txt')
z = np.array(z)
z = z.transpose()
plt.figure()
im = plt.imshow(z, interpolation='bilinear', origin='lower',
	aspect='auto', extent=(0, times[-1], -45, 45), cmap=cmap)
CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)
plt.xlabel('Time, ms')
plt.ylabel('z, $\mu$m')
plt.savefig('axial_pi_pulse.pdf')
