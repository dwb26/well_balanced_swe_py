import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from matplotlib import animation

curve_data_f = open("curve_data.txt", "r")
Z_data_f = open("Z_data.txt", "r")
times_f = open("times.txt", "r")
hmm_data_f = open("hmm_data.txt", "r")

length = 5
nx = 200
space_left = 1.0; space_right = 26.0
thetas = []; obs = []
h = []
for line in curve_data_f:
	h.extend(list(map(float, line.split())))
total_length = int(h[-1])

times = np.array(list(map(float, times_f.readline().split()))) * 100
minutes = np.zeros(total_length)
n_mins = 0
for n in range(1, total_length):
	minutes[n] = n_mins
	if (times[n - 1] % 60) > (times[n] % 60):
		n_mins += 1

thetas = list(map(float, hmm_data_f.readline().split()))
# for n in range(total_length):
	# x, y = list(map(float, hmm_data_f.readline().split()))
	# x = float(hmm_data_f.readline())
	# thetas.append(x);
	# obs.append(y)

h = h[:-1]
h_arr = np.array(h).reshape((total_length, nx))
Z_arr = np.empty((total_length, nx))
m = 0
for line in Z_data_f:
	Z_arr[m] = list(map(float, line.split()))
	m += 1

fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
xs = np.linspace(space_left, space_right, nx)
line1, = ax.plot(xs, h_arr[0], "dodgerblue")
line2, = ax.plot(xs, Z_arr[0], "black")

def update(n):
	line1.set_data(xs, h_arr[n])
	line2.set_data(xs, Z_arr[n])
	ax.set_title(r"Time = {}:{:.2f}, $\theta$ = {:.2f}".format(int(minutes[n]), times[n] % 60, thetas[n]))
	# ax.set_title("Time = {:.2f}".format(times[n]))
	ax.set(ylim=np.max(h_arr) + 0.1)
	ax.set(ylim=np.min(Z_arr) - 0.1)
	return line1, line2,

ani = animation.FuncAnimation(fig, func=update, frames=range(0, total_length, 5))
plt.show()