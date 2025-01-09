"""
This file is to learn and imitate the original process of patient data processing.
(1) From raw data to data input
Source: HERCULES/data_import/importUbisense.ipynb

Note: the grouped data produced here is not stored. The corresponding file is in input_to_grouped.py.py.
"""


'''
Initialisation
'''

import pandas as pd

order_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
order_list_noweekend = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

pd.set_option('display.max_rows', 300)  # specifies number of rows to show
# pd.set_option('display.max_columns', 10)  # specifies number of rows to show
pd.options.display.float_format = '{:40,.4f}'.format # specifies default number format to 4 decimal places

'''
Edit fields below before processing data
'''

phase = 'P4_staff_2023_02' # edit which Phase you are analysing - this is used in graph and file generation
# this script assumes CSV above has MM/DD/YYYY format - if not changes needed in next section below

start_date = '2023-02-01' # edit these for reducing processed download between 2 dates
end_date = '2023-03-01'

df = pd.read_csv('tech/rawdata/p4_tech_2023_02.csv')

'''
Data cleaning and formatting
'''

df = df.dropna() # remove any rows with null values
df = df.dropna() # remove any rows with null values
df[['xlocation', 'ylocation']] = df['Location'].str.split(',', expand = True) # create seperate columns for the x y values
df.rename(columns={"from": "starttime", "to": "endtime"}, inplace=True) # renaming from and to column headings (from is a keyword)
df['starttime'] = pd.to_datetime(df['starttime']) #, dayfirst=True)
df['endtime'] = pd.to_datetime(df['endtime']) #, dayfirst=True)

df.rename(columns={'Technician':'Staff'}, inplace=True)
# look for any records that don't (~) start with S and then drop those rows
df = df.drop(df[~df["Staff"].str.startswith('S')].index)

df['step_length'] = df['endtime'] - df['starttime'] # add in variable that reports the time at each step between records
'''
In the original file for patient data, all traces lasting more than 2 hours are deleted using the following method.
But here we do not do this in staff data at the moment.
There are records with a duration of 0 seconds, corresponding to spatial flashes. They should be removed.
'''
df = df.drop(df[df['step_length'] <= pd.Timedelta(0, 'h')].index)

# check the start and end dates of the phase being reported yyyy-mmm-dd
# This part can be used again after the following data cleaning to double-check.
# print("Earliest Date: ", df.starttime.min())
# print("Latest Date:   ", df.endtime.max())

# ensure the data is within the date range
mask = (df['starttime'] > start_date) & (df['endtime'] < end_date)
df = df.loc[mask]

# remove overnight records
df = df.loc[df['starttime'].dt.date == df['endtime'].dt.date]

df.sort_values(by=['Staff', 'starttime'], inplace=True)

# remove records beyond the normal working time
df = df.loc[df['starttime'].dt.time > pd.to_datetime('9:00:00').time()]
df = df.loc[df['endtime'].dt.time < pd.to_datetime('18:00:00').time()]

df.reset_index(drop=True, inplace=True)

# print(df)

'''
Create a new datafame to group each patient id.
Since staff appear in multiple days, the treatment should be different from the patient data.
For each staff member, the trace should be separated by dates.
'''

df['date'] = df['starttime'].dt.date  # Add a new column for grouping.
dfgrouped = df.groupby(['Staff', 'date'], as_index=False).agg(
    {'starttime': ['min'], 'endtime': ['max'], 'xlocation': ['first'], 'ylocation': ['first']}
)

# Now the dfgrouped has two layers of column titles, and the following code is for flattening them.
flat_cols = []
for i in dfgrouped.columns:
    flat_cols.append(i[0]) # take the first element of the column heading only (ie ignore min, max, first)
dfgrouped.columns = flat_cols

dfgrouped['work_length'] = dfgrouped['endtime'] - dfgrouped['starttime']


'''
Export back to clean csv
'''
df.drop('date', axis=1, inplace=True)
df.to_csv('tech/Input/{}_input.csv'.format(phase), index=False)
dfgrouped.to_csv('tech/Grouped/{}_grouped_data.csv'.format(phase), index=False)

# print(dfgrouped)
