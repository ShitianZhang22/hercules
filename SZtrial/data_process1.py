"""
This is for converting the original .csv staff data into all-digit .txt file for numpy.

Note: some lines have missing data
"""

import numpy as np
import pandas
import os


def date_transfer(d_str=''):
    d_str = d_str.split('/')
    for j in range(2):
        if len(d_str[j]) < 2:
            d_str[j] = '0' + d_str[j]
    return int(d_str[2] + d_str[0] + d_str[1])


def time_transfer(t_str=''):
    t_str = t_str.split(' ')
    _temp = t_str[1].split(':')
    _temp1 = int(_temp[0]) * 3600 + int(_temp[1]) * 60 + int(_temp[2])
    if t_str[2] == 'PM' and _temp[0] != '12':  # there are forms like 12:01 PM in the data
        _temp1 += 43200
    return _temp1


if not os.path.exists('tech'):
    os.makedirs('tech')

file0 = 'p4_tech_2023_02'
file = r'../data/live/ubisense_rawdata/technician/' + file0 + '.csv'
data = pandas.read_csv(file)
data2 = np.zeros((data.shape[0], 8))  # series, staff, date, x, y, from, to, duration

series = -1
staff = None
date = None
for i in range(data.shape[0]):
    print(i)
    switch = False  # indicator for the next series
    temp = data.iloc[i]

    # staff
    temp1 = temp.iloc[0].split(' ')[1]
    if staff != temp1:
        staff = temp1
        switch = True
    # date
    temp1 = date_transfer(temp.iloc[2].split(',')[0])
    if date != temp1:
        date = temp1
        switch = True
    # # break at long duration
    # if int(temp[-1]) > 10000:
    #     switch = True
    if switch:
        series += 1
    data2[i, 0] = series
    data2[i, 1] = staff
    data2[i, 2] = date
    data2[i, 3], data2[i, 4] = temp.iloc[1].split(',')  # xy position
    data2[i, 5] = time_transfer(temp.iloc[2])
    data2[i, 6] = time_transfer(temp.iloc[3])
    data2[i, 7] = data2[i, 6] - data2[i, 5]
    if data2[i, 7] < 0:  # date change
        data2[i, 7] = 0
np.savetxt(r'tech/converted/' + file0 + '.csv', data2, fmt='%f', delimiter=',', encoding='utf-8')

'''
note:
data in 2022_10 D110210 is missing. Copied from C110210.
data in 2022_11 D99428
data in 2023_02 D33280
data in 2023_02 D302980
data in 2023_02 D319699
'''