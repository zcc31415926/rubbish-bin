import matplotlib.pyplot as plt
from pylab import *
import numpy as np

num_curves = 6
num_columns = 1
num_samples = 81

bottom_left_cross = [0, 0]
top_right_cross = [800, 0.3]
measurement_value = [100, 0.1]

# font size of label, ticks and legend
font_size_title = 20
font_size_label = 15
font_size_tick = 15
font_size_legend = 15

# input: x and y
x = 10 * np.array([i for i in range(num_samples)])
y = np.zeros([num_curves, num_samples])
y[0] = np.loadtxt('/home/charlie/Desktop/pics-depool/t2_comparison_operation/net1/loss.txt')
y[1] = np.loadtxt('/home/charlie/Desktop/pics-depool/t2_comparison_operation/net2/loss.txt')
y[2] = np.loadtxt('/home/charlie/Desktop/pics-depool/t2_comparison_operation/net3/loss.txt')
y[3] = np.loadtxt('/home/charlie/Desktop/pics-depool/t2_comparison_operation/net4/loss.txt')
y[4] = np.loadtxt('/home/charlie/Desktop/pics-depool/t2_comparison_operation/net5/loss.txt')
y[5] = np.loadtxt('/home/charlie/Desktop/pics-depool/t2_comparison_operation/net6/loss.txt')

# curve attributes
color = ['#000000', '#00FF00', '#FF0000', '#000000', '#00FF00', '#FF0000']
linestyle = ['-', '-', '-', '--', '--', '--']
label = ['conv1', 'actv1', 'pool1', 'conv2', 'actv2', 'pool2']

axes=plt.gca()

# axes visibility and cross position
# axes.spines['top'].set_visible(False)
# axes.spines['right'].set_visible(False)
axes.spines['left'].set_position(('data', bottom_left_cross[0]))
axes.spines['bottom'].set_position(('data', bottom_left_cross[1]))
axes.spines['right'].set_position(('data', top_right_cross[0]))
axes.spines['top'].set_position(('data', top_right_cross[1]))

axes_righty = None
axes_topx = None
# axes_righty = axes.twinx()
# axes_topx = axes.twiny()

# title and label names
# plt.title('Loss Convergence Processes with Strict Manipulation of Different Outputs.', fontsize=font_size_title)
axes.set_xlabel('Number of iterations', fontsize=font_size_label)
axes.set_ylabel('Loss value', fontsize=font_size_label)
if axes_topx != None:
    axes_topx.set_xlabel('Number of iterations', fontsize=font_size_label)
if axes_righty != None:
    axes_righty.set_ylabel('Classification confidence - 3', fontsize=font_size_label)

# axes trimmimg (length)
xlim([bottom_left_cross[0], top_right_cross[0]])
ylim([bottom_left_cross[1], top_right_cross[1]])

# axis ticks
x_ticks = np.arange(bottom_left_cross[0], top_right_cross[0] + 1e-4, measurement_value[0])
y_ticks = np.arange(bottom_left_cross[1], top_right_cross[1] + 1e-4, measurement_value[1])
axes.set_xticks(x_ticks)
axes.set_yticks(y_ticks)
for tick in axes.xaxis.get_majorticklabels():
    tick.set_fontsize(font_size_tick)
for tick in axes.yaxis.get_majorticklabels():
    tick.set_fontsize(font_size_tick)
if axes_topx != None:
    axes_topx.set_xticks(x_ticks)
    for tick in axes_topx.xaxis.get_majorticklabels():
        tick.set_fontsize(font_size_tick)
if axes_righty != None:
    axes_righty.set_yticks(y_ticks)
    for tick in axes_righty.yaxis.get_majorticklabels():
        tick.set_fontsize(font_size_tick)

# curves with color, line style and name settings
for i in range(num_curves):
    plt.plot(x, y[i], color=color[i], linestyle=linestyle[i], label=label[i])

# others
plt.legend(fontsize=font_size_legend, ncol=num_columns)
# plt.grid(axis='x')
plt.grid()

plt.show()
