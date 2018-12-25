import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
import time
import numpy as np
from utils import Logger

plt.ion()

class ShowVariableLogger(Logger):
    def __init__(self, average_window=1, update_frequency=1):
        self.update_period = 1 / update_frequency
        self.average_window = average_window
        self.last_update_time = {}
        self.values = {}
        self.ax = {}

    def _init_var_graph(self, var):
        self.last_update_time[var] = time.time()
        self.values[var] = []
        fig = plt.figure()
        fig.canvas.set_window_title('Evolution of '+var)
        self.ax[var] = fig.add_subplot(1, 1, 1)
        self.ax[var].set_xlim(auto=True)
        self.ax[var].set_ylim(auto=True)
        plt.draw()
        plt.pause(1e-5)

    def _update_var_value(self, var, value):
        buffer_size = len(self.values[var])
        if self.average_window == 1 or buffer_size == 0:
            new_value = value
        elif buffer_size < self.average_window:
            new_value = (self.values[var][-1] * buffer_size + value) / (buffer_size + 1)
        else:
            new_value = self.values[var][-1] + (value - self.values[var][-self.average_window]) / self.average_window
        self.values[var].append(new_value)

    def log(self, var, value):
        if var not in self.last_update_time:
            self._init_var_graph(var)
        self._update_var_value(var, value)
        if time.time() > self.last_update_time[var] + self.update_period:
            self.last_update_time[var] = time.time()
            self._update_var_graph(var)

    def _update_var_graph(self, var):
        ax = self.ax[var]
        values = self.values[var]
        ax.clear()
        ax.set_title('Evolution of Variable ' + var)
        size = len(values)
        if size < 200:
            ax.plot(range(size), values)
        else:
            i_range = range(0, size, int(size/200))
            ax.plot(i_range, np.take(values, i_range))
        plt.draw()
        plt.pause(1e-5)

