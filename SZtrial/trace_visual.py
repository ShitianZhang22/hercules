"""
This file is for visualising the staff traces.
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.collections import LineCollection

"""
loading file
cols: series, staff, date, x, y, from, to, duration
"""

file0 = '2022_10'
file = r'tech/converted/' + file0 + '.csv'
data = np.loadtxt(file, dtype='float', delimiter=',', encoding='utf-8')

"""
visualisation
"""
print(data[:, 5].max())
print(data[:, 5].min())

# floor plan
img = Image.open('../data_vis_pde/load_data/img/phase4.png')
img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def draw(_id):
    print(_id)
    sample = data[data[:, 0] == _id]

    # let the coordinate fit the size of pictures
    x = sample[:, 3] * 1200 / 34.441
    y = sample[:, 4] * 635 / 18.209

    # Plot the data:

    '''
    The following part is for rendering the traces according to time.
    '''
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, 24)
    lc = LineCollection(segments, cmap='inferno_r', norm=norm)
    # Set the values used for color mapping
    lc.set_array(sample[: sample.shape[0]-1 , 5] / 3600)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    # set colorbar
    cbar = fig.colorbar(line, ax=ax, ticks=[0, 4, 8, 12, 16, 20, 24], label='Time')
    cbar.ax.set_yticklabels(['0am', '4am', '8am', '12pm', '4pm', '8pm', '0am'])


    ax.set_xlim([0, 1200])
    ax.set_ylim([0, 635])
    ax.imshow(img, origin='upper')

    _from, _to = sec_to_hr(int(sample[0, 5])), sec_to_hr(int(sample[-1, 5]))
    ax.set_title("Staff {}  in {}\nFrom: {}   To: {}\nDuration: {:.2f} hours".format(
        int(sample[0, 1]), int(sample[0, 2]),_from ,_to , duration(_id)))
    # plt.show()

    # save image
    plt.savefig('tech/pic/{}/{}.png'.format(file0, _id))
    plt.close()


def std_error():
    """
    This is for calculating the standard error of x and y coordinates for all traces
    :param :
    :return: N * 2 array for all x and y std error
    """
    std_x, std_y = [], []
    for i in range(int(data[-1, 0])):
        sample = data[data[:, 0] == i]
        std_x.append(np.std(sample[:, 3]))
        std_y.append(np.std(sample[:, 4]))

    '''
    The following part is for rendering the points according to time.
    '''

    fig, ax = plt.subplots(figsize=(10, 5))

    # calculating duration of each traces
    dur = []
    for i in range(int(data[-1, 0])):
        dur.append(duration(i))

    scatter = ax.scatter(std_x, std_y, c=dur, cmap='viridis_r')
    ax.set_xlabel('Standard deviation of X')
    ax.set_ylabel('Standard deviation of Y')
    ax.grid(True)

    # legend
    legend = ax.legend(*scatter.legend_elements(num=5), title='Duration')
    ax.add_artist(legend)

    plt.show()
    plt.close()


def duration(_id):
    """
    This is for summarising the durations of all tests.
    :param _id: The ID of the trace.
    :return:
    """
    sample = data[data[:, 0] == _id]
    temp = (sample[-1, 5] - sample[0, 5]) / 3600
    return temp

def sec_to_hr(_t):
    """
    This is for converting time format from seconds to HH:MM:SS
    :param _t: time in seconds
    :return: time in HH:MM:SS
    """
    temp = _t // 3600
    if temp < 10:
        temp = '0' + str(temp)
    else:
        temp = str(temp)
    out = temp + ':'
    _t %= 3600
    temp = _t // 60
    if temp < 10:
        temp = '0' + str(temp)
    else:
        temp = str(temp)
    out += temp + ':'
    temp = _t % 60
    if temp < 10:
        temp = '0' + str(temp)
    else:
        temp = str(temp)
    return out + temp

'''
drawing images
'''
for i in range(int(data[-1, 0])):
    draw(i)

# Below is a single example
# draw(2)

'''
calculating std error
'''

# std_error()

'''
histogram of durations
'''
# for i in range(int(data[-1, 0])):
#     duration(i)
# plt.hist(dur)
# plt.title('Durations of Traces')
# plt.xlabel('Durations (h)')
# plt.show()


img.close()
