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

no_noise_x, no_noise_y = load_xy_data('visibility_150k_no_noise.txt')
noise_x1, noise_y1 = load_xy_data('visibility_150k_noise_16_16_128_4ens.txt')
noise_x2, noise_y2 = load_xy_data('visibility_150k_noise_32_32_128_4ens.txt')
noise_x3, noise_y3 = load_xy_data('visibility_150k_noise_16_16_128_16ens.txt')

plt.figure()
plt.plot(no_noise_x, no_noise_y, label='No quantum noise')
plt.plot(noise_x1, noise_y1, label='Noise, 16x16x128 lattice, 4 ensembles')
plt.plot(noise_x2, noise_y2, label='Noise, 32x32x128 lattice, 4 ensembles')
plt.plot(noise_x3, noise_y3, label='Noise, 16x16x128 lattice, 16 ensembles')
plt.legend()
plt.xlabel('Time, ms')
plt.ylabel('Visibility')
plt.axis([0, 600, 0, 1])
plt.grid(True)
plt.savefig('visibility_150k.pdf')

no_noise_x, no_noise_y = load_xy_data('visibility_70k_no_noise.txt')
noise_x1, noise_y1 = load_xy_data('visibility_70k_noise_16_16_128_4ens.txt')

plt.figure()
plt.plot(no_noise_x, no_noise_y, label='No quantum noise')
plt.plot(noise_x1, noise_y1, label='With quantum noise, 4 ensembles')
plt.legend()
plt.xlabel('Time, ms')
plt.ylabel('Visibility')
plt.axis([0, 600, 0, 1])
plt.grid(True)
plt.savefig('visibility_70k.pdf')

exit(0)
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
