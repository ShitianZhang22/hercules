"""
This file is to learn and imitate the original process of patient data processing.
(2) From data input to grouped data
Source: HERCULES/data_processing/journey_stats.ipynb
"""

'''
Initialisation
'''
import pandas as pd
import numpy as np
import datetime as dt

order_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
order_list_noweekend = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

'''
Import data. Check this section before every run of data processing.
'''

# read in CSV data
df = pd.read_csv('C:/Users/zst18/Documents/project/hercules/SZtrial/tech/rawdata/p4_tech_2022_09.csv')
# convert dates into ones that can be used in pandas
df['starttime'] = pd.to_datetime(df['starttime']) # datetime in YYYY-MM-DD HH:MM:SS format
df['endtime'] = pd.to_datetime(df['endtime'])

df.sort_values(by='starttime', inplace = True)
print(df)

