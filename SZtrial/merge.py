'''
This file is for merging all staff data files.
'''

import pandas as pd
import os

merge_dir = 'tech/Input/Merged_input.csv'
if os.path.exists(merge_dir):
    os.remove(merge_dir)

files = [
    '2022_09',
    '2022_10',
    '2022_11',
    '2022_12',
    '2023_02',
]
for i in range(len(files)):
    df = pd.read_csv('tech/Input/P4_staff_{}_input.csv'.format(files[i]))

    # The original type of the following locations is object. They should be converted to float64.
    df['x_location'] = pd.to_numeric(df['x_location'])
    df['y_location'] = pd.to_numeric(df['y_location'])
    # tables except for the first one do not need headers!
    _h = False
    if i == 0:
        _h = True
    df.to_csv(merge_dir, mode='a', index=False, date_format='%Y-%m-%d %H:%M:%S', encoding='utf-8', header=_h)
