import PIL.ImageOps
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


"""
loading file
cols: series, staff, date, x, y, from, to, duration
"""

file0 = 'p4_tech_2022_09'
file = r'tech/' + file0 + '.txt'
data = np.loadtxt(file, dtype='float', delimiter=',', encoding='utf-8')

"""
visualisation
"""
# the id of trace to display
trace_id = 5
sample = data[data[:, 0] == trace_id]
print(sample)
print(sample.shape)

img = Image.open('../data_vis_pde/load_data/img/phase4.png')
img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

# let the coordinate fit the size of pictures
sample[:, 3] = sample[:, 3] * 1200 / 34.441
sample[:, 4] = sample[:, 4] * 635 / 18.209

# Plot the data:
plt.figure(figsize=(10, 6))
plt.plot(sample[:, 3], sample[:, 4])
# Set x,y lower, upper limits:
# plt.xlim([0, 34.441])
# plt.ylim([0, 18.209])
plt.xlim([0, 1200])
plt.ylim([0, 635])
plt.imshow(img, origin='upper')
plt.title("Staff " + str(int(sample[0, 1])))
plt.show()
img.close()
