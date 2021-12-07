import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# hmm_f = open("hmm_data.txt", "r")
data = open("curve_data.txt", "r")

# length = int(hmm_f.readline())
# signal = np.empty(length)
# for i in range(2):
# 	hmm_f.readline().split()
# nx = int(hmm_f.readline())
# for i in range(4):
# 	hmm_f.readline().split()
# for i in range(length):
# 	signal[i] = list(map(float, hmm_f.readline().split()))[0]

nx = 200
length = 1
m = 0; n = 0
# curves = []
h = []; u = []
counters = np.zeros(length, dtype=int)

for line in data:
	if m % 2 == 0:
		h.extend(list(map(float, line.split())))
	else:
		u.extend(list(map(float, line.split())))
		# counters[n] = list(map(int, line.split()))[0]
		# n += 1
	m += 1
counters[0] = int(h[-1])
h = h[:-1]
total_length = np.sum(counters)
h_arr = np.array(h).reshape((total_length, nx))
u_arr = np.array(u).reshape((total_length, nx))
# curve_arr = np.array(curves).reshape((total_length, nx))
def gen_Z_drain(x):
	Z = np.empty(len(x))
	for i in range(len(x)):
		Z[i] = np.max((0.2 - 0.05 * (x[i] - 10) ** 2, 0))
	return Z

fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
xs = np.linspace(0, 25, nx)
Z_drain = gen_Z_drain(xs)
# line, = ax.plot(xs, curve_arr[0])
line1, = ax1.plot(xs, h_arr[0])
top, = ax1.plot(xs, Z_drain)
line2, = ax2.plot(xs, h_arr[0])

def update(n):
	# line.set_data(xs, curve_arr[n])
	line1.set_data(xs, h_arr[n])
	# top.set_data(xs, h_arr[n])
	line2.set_data(xs, u_arr[n])
	ax1.set_title("iterate = {} / {}".format(n, total_length))
	# ax.set(ylim=(np.min(curve_arr), np.max(curve_arr)))
	ax1.set(ylim=(np.min(h_arr) - 1, np.max(h_arr) + 1))
	ax2.set(ylim=(np.min(u_arr), np.max(u_arr)))
	return line1, line2,

ani = animation.FuncAnimation(fig, func=update, frames=range(1, total_length, 5))
plt.show()