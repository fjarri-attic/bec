import matplotlib.pyplot as plt
import sys

f = open(sys.argv[1])
times = []
values = []
for line in f:
    elems = line.split(' ')
    times.append(float(elems[0]))
    values.append(float(elems[1]))

plt.plot(times, values)
plt.xlabel('Time, ms')
plt.ylabel('Visibility')
plt.axis([0, 600, 0, 1])
plt.grid(True)
plt.savefig(sys.argv[2])

