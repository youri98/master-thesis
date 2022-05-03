import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from math import floor

def hhmmss_formatter(x, pos=None):
    s = floor(x % 60)
    m = floor((x - s) / 60) % 60
    h = floor(x / 3600)
    return f'{h}:{m:02d}' if s == 0 else f'{h}:{m:02d}:{s:02d}'

def show_annotation(sel):
    xi = sel.target[0]
    vertical_line = ax.axvline(xi, color='red', ls=':', lw=1)
    sel.extras.append(vertical_line)
    # print('label:', sel.artist.get_label())
    y1i = np.interp(xi, x, y)
    y2i = np.interp(xi, x, y2)
    y3i = np.interp(xi, x, y3)
    annotation_str = f'Time: {hhmmss_formatter(xi)}\nVoltage: {y1i:.1f}\nCurrent: {y2i:.1f}\nTemperature: {y3i:.1f}'
    sel.annotation.set_text(annotation_str)

x = np.linspace(0, 85000, 500)
y = np.random.randn(len(x)).cumsum() / 5 + 15
y2 = np.random.randn(len(x)).cumsum() / 5 + 30
y3 = np.random.randn(len(x)).cumsum() / 5 + x / 10000 + 25

fig = plt.figure(figsize=(13, 5))

ax = fig.add_subplot(111)

ax.plot(x, y, "--", label="Voltage")
ax.plot(x, y2, "-.", label="Current")

ax2 = ax.twinx()
ax2.plot(x, y3, "g:", label="Temperature")
ax2.set_ylabel("Celsius", color="LightBlue")
ax2.set_ylim(18, 100)

fig.legend(edgecolor=("DarkBlue"), facecolor=("LightBlue"), loc="upper right", bbox_to_anchor=(1, 1),
           bbox_transform=ax.transAxes, )

ax.set_title("Test Surveillance", color="Purple")
ax.set_xlabel("Seconds", color="LightGreen")
ax.set_ylabel("Volts and Amps", color="DarkGrey")

ax.set_xlim(xmin=0)
ax.xaxis.set_major_formatter(plt.FuncFormatter(hhmmss_formatter))  # show x-axis as hh:mm:ss
ax.xaxis.set_major_locator(plt.MultipleLocator(2 * 60 * 60))  # set ticks every two hours

cursor = mplcursors.cursor(hover=True)

cursor.connect('add', show_annotation)
#fig.canvas.mpl_connect('add', show_annotation)

plt.show()
