import matplotlib.pyplot as plt
import numpy as np


bottom_left = [0, 0]
top_right = [10, 30]
tick_len = [1, 10]

# font size of label, ticks and legend
fontsize_title = 20
fontsize_label = 15
fontsize_tick = 15
fontsize_legend = 15

# 4 lines. 10 points on each line
x = np.array([i * 10 for i in range(10)])
y = np.zeros([4, 10])
for i in range(4):
    y[i] = 0.5 * (i + 1) * x

# curve attributes
color = ['#0000FF', '#00FF00', '#FF0000', '#000000']
linestyle = ['-', '-.', ':', '--']
label = [f'line{i}' for i in range(4)]
width = [1, 2, 3, 4]

axes = plt.gca()

# axes visibility and crossing position
# axes.spines['top'].set_visible(False)
# axes.spines['right'].set_visible(False)
axes.spines['left'].set_position(('data', bottom_left[0]))
axes.spines['bottom'].set_position(('data', bottom_left[1]))
axes.spines['right'].set_position(('data', top_right[0]))
axes.spines['top'].set_position(('data', top_right[1]))

axes_righty = None
axes_topx = None
# axes_righty = axes.twinx()
# axes_topx = axes.twiny()

# title and label names
plt.title('Figure', fontsize=fontsize_title)
axes.set_xlabel('X axis', fontsize=fontsize_label)
axes.set_ylabel('Y axis', fontsize=fontsize_label)
if axes_topx != None:
    axes_topx.set_xlabel('X axis on the top', fontsize=fontsize_label)
if axes_righty != None:
    axes_righty.set_ylabel('Y axis on the right', fontsize=fontsize_label)

# axes trimming (length)
plt.xlim([bottom_left[0], top_right[0]])
plt.ylim([bottom_left[1], top_right[1]])

# axis ticks
x_ticks = np.arange(bottom_left[0], top_right[0] + 1e-4, tick_len[0])
y_ticks = np.arange(bottom_left[1], top_right[1] + 1e-4, tick_len[1])
axes.set_xticks(x_ticks)
axes.set_yticks(y_ticks)
for tick in axes.xaxis.get_majorticklabels():
    tick.set_fontsize(fontsize_tick)
for tick in axes.yaxis.get_majorticklabels():
    tick.set_fontsize(fontsize_tick)
if axes_topx != None:
    axes_topx.set_xticks(x_ticks)
    for tick in axes_topx.xaxis.get_majorticklabels():
        tick.set_fontsize(fontsize_tick)
if axes_righty != None:
    axes_righty.set_yticks(y_ticks)
    for tick in axes_righty.yaxis.get_majorticklabels():
        tick.set_fontsize(fontsize_tick)

# curves with color, line style and name settings
for i in range(4):
    plt.plot(x, y[i], color=color[i], linestyle=linestyle[i], label=label[i], linewidth=width[i])

# others
plt.legend(fontsize=fontsize_legend, ncol=1)
# plt.grid(axis='x')
# plt.grid()

plt.show()

