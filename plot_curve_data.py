import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

data = open("curve_data.txt", "r")
times_f = open("times.txt", "r")
times = list(map(float, times_f.readline().split()))

nx = 200
length = 1
m = 0; n = 0
h = []; q = []
space_left = 0.0; space_right = 50.0
centre = 10; height = 0.2
inc_start = 12 + 10
land_height = 0.2
counters = np.zeros(length, dtype=int)

for line in data:
	if m % 2 == 0:
		h.extend(list(map(float, line.split())))
	else:
		q.extend(list(map(float, line.split())))
	m += 1
counters[0] = int(h[-1])
h = h[:-1]
total_length = np.sum(counters)
h_arr = np.array(h).reshape((total_length, nx))
q_arr = np.array(q).reshape((total_length, nx))
def gen_Z_drain(x):
	Z = np.empty(len(x))
	for i in range(len(x)):
		Z[i] = np.max((0.2 - 0.05 * (x[i] - 10) ** 2, 0))
		# Z[i] = np.min((-(0.2 - 0.05 * (x[i] - 10) ** 2), 0))
	return Z
def gen_Z_step(x):
	Z = np.empty(len(x))
	a = 25 / 3; b = 25 / 2
	for i in range(len(x)):
		if x[i] < a or x[i] > b:
			Z[i] = 0
		else:
			Z[i] = 1
	return Z
def gen_Z_coast(x):
	Z = np.empty(len(x))
	for i in range(len(x)):
		Z[i] = np.max((height - 0.05 * (x[i] - centre) ** 2, 0)) # 10 is the centre, height is 0.2
	m = 0.5 * np.pi / (space_right - inc_start)
	for i in range(len(x)):
		if x[i] >= inc_start:
			Z[i] = land_height * np.sin(m * (x[i] - inc_start))
	return Z

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
xs = np.linspace(space_left, space_right, nx)
Z_drain = gen_Z_drain(xs)
Z_step = gen_Z_step(xs)
Z_coast = gen_Z_coast(xs)
line1, = ax1.plot(xs, h_arr[0])
line2, = ax2.plot(xs, q_arr[0])
# top, = ax1.plot(xs, Z_drain, "k")
# top, = ax1.plot(xs, Z_step, "k")
top, = ax1.plot(xs, Z_coast, "k")

def update(n):
	line1.set_data(xs, h_arr[n])
	line2.set_data(xs, q_arr[n])
	# top.set_data(xs, Z_drain)
	# top.set_data(xs, Z_step)
	top.set_data(xs, Z_coast)
	ax1.set_title("Time = {:.2f}".format(times[n]))
	ax1.set(ylim=(-0.1, np.max(h_arr) + 0.1))
	# ax1.set(ylim=(0, 0.55))
	ax2.set(ylim=(np.min(q_arr) - 0.1, np.max(q_arr) + 0.1))
	# return line1, line2,
	return line1, line2, top,	

ani = animation.FuncAnimation(fig, func=update, frames=range(0, total_length, 5))
plt.show()