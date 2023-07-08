import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib import animation as animation
animation.ffmpeg_path = '/path/to/ffmpeg'
matplotlib.use('TkAgg')  # Replace 'TkAgg' with an appropriate backend (e.g., 'Qt5Agg', 'Agg')
animation.writer = 'pillow'
import os
current_dir = os.getcwd()
print(current_dir)
####1
mean = [1, 1]
covariance_matrix = [[1, 0.2], [0.2, 0.8]]
sample_size = 10000
np.random.seed(0)
sample = np.random.multivariate_normal(mean, covariance_matrix, sample_size)
x = sample[:, 0]
y = sample[:, 1]
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].scatter(x, y, alpha=0.5)
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('X2')
axs[0, 0].set_title('Scatter Plot')
axs[1, 0].hist(x, bins=30, density=True, alpha=0.5, color='blue')
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('Density')
axs[1, 0].set_title('Marginal Distribution of X')
axs[0, 1].hist(y, bins=30, density=True, alpha=0.5, color='green', orientation='horizontal')
axs[0, 1].set_xlabel('Density')
axs[0, 1].set_ylabel('X2')
axs[0, 1].set_title('Marginal Distribution of X2')
x_range = np.linspace(-2, 4, 100)
x_pdf = multivariate_normal.pdf(x_range, mean[0], covariance_matrix[0][0])
axs[1, 0].plot(x_range, x_pdf, color='red', label='Analytical PDF')
y_range = np.linspace(-2, 4, 100)
y_pdf = multivariate_normal.pdf(y_range, mean[1], covariance_matrix[1][1])
axs[0, 1].plot(y_pdf, y_range, color='red', label='Analytical PDF')
# Legend
axs[1, 0].legend()
axs[0, 1].legend()


plt.tight_layout()


plt.show()

####2


def f(x):
    return 0.25 * (x + 4) * (x + 1) * (x - 2)


x_min, x_max = -4, 4
x_range = np.linspace(x_min, x_max, 1000)


y = f(x_range)


min_index = np.argmin(y)
x_min = x_range[min_index]
y_min = y[min_index]


fig = plt.figure(figsize=(1280/72, 720/72))  # 1280x720 pixels, 72 dpi
fig.canvas.toolbar.pack_forget()  # Disable toolbar


ax = fig.add_subplot(111)
ax.plot(x_range, y, color='blue', linewidth=2)


ax.annotate('Min', xy=(x_min, y_min), xytext=(x_min+0.5, y_min+2), fontsize=12,
            arrowprops=dict(facecolor='black', shrink=0.05))


ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Plot of f(x) = 0.25 * (x + 4)(x + 1)(x - 2)')


plt.show()
plt.savefig('plot.png')

ax_sub = ax.inset_axes([0.5, 0.2, 0.35, 0.35])
ax_sub.plot(x_range, y, color='blue', linewidth=2)
ax_sub.set_xlim(-1, 2)
ax_sub.set_ylim(-2, 2)

dot, = ax.plot([], [], color='red', marker='o', markersize=8)



def init():
    dot.set_data([], [])
    return dot,


def update(frame):
    x = [x_range[frame]]  # Wrap x in a list
    y = [f(x[0])]  # Wrap y in a list
    dot.set_data(x, y)
    return dot,

ani = animation.FuncAnimation(fig, update, frames=len(x_range), init_func=init, blit=True)

# ı cant save as mp4 because ffmpeg an imafemagick is not available my version and ı cant solve so ı have used pillow and gif format.
ani.save('animation.gif', writer='pillow')

print("Animation saved as animation.gif")

