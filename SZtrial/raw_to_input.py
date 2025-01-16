"""
This file is to learn and imitate the original process of patient data processing.
(1) From raw data to data input
Source: HERCULES/data_import/importUbisense.ipynb
(2) From data input to grouped data and statistical analyses
Source: HERCULES/data_processing/journey_stats.ipynb
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

fig_size = (16,10) # how big the exported figures should be - width, height - in inches!
ymax_boxplot = 240 # max scale on the box plots to normalise across phases

'''
Edit fields below before processing data
'''

print('Have you checked the raw data path?')

# store information about different tables
current = '2023_02'
config = {
    '2022_09': ['2022_09', '2022-09-01', '2022-10-01', '2022_09'],
    '2022_10': ['2022_10', '2022-10-01', '2022-11-01', '2022_10'],
    '2022_11': ['2022_11', '2022-11-01', '2022-12-01', '2022_11'],
    '2022_12': ['2022_12', '2022-12-01', '2023-01-01', '2022_12'],
    '2023_02': ['2023_02', '2023-02-01', '2023-03-01', '2023_02'],
}

phase = 'P4_staff_' + config[current][0] # edit which Phase you are analysing - this is used in graph and file generation
# this script assumes CSV above has MM/DD/YYYY format - if not changes needed in next section below

start_date = config[current][1] # edit these for reducing processed download between 2 dates
end_date = config[current][2]

df = pd.read_csv('tech/rawdata/p4_tech_{}.csv'.format(config[current][3]))

'''
Data cleaning and formatting
'''

df = df.dropna() # remove any rows with null values

# change the staff data format to be consistent with the database
for i in range(len(df)):
    temp = df.iloc[i, 0].split(' ')[1]  # extract id
    df.iloc[i, 0] = 'S' + '0' * (4 - len(temp)) + temp

# remane columns to be consistent with the database
df.rename(columns={
    'Location': 'location',
    'from': 'start_time',
    'to': 'end_time',
    }, inplace=True)

df[['x_location', 'y_location']] = df['location'].str.split(',', expand = True) # create seperate columns for the x y values
df['start_time'] = pd.to_datetime(df['start_time']) #, dayfirst=True)
df['end_time'] = pd.to_datetime(df['end_time']) #, dayfirst=True)

df.rename(columns={'Technician':'patient_id'}, inplace=True)
# look for any records that don't (~) start with S and then drop those rows
df = df.drop(df[~df["patient_id"].str.startswith('S')].index)

df['step_length'] = df['end_time'] - df['start_time'] # add in variable that reports the time at each step between records
'''
In the original file for patient data, all traces lasting more than 2 hours are deleted using the following method.
But here we do not do this in staff data at the moment.
There are records with a duration of 0 seconds, corresponding to spatial flashes. They should be removed.
'''
df = df.drop(df[df['step_length'] <= pd.Timedelta(0, 'h')].index)

# check the start and end dates of the phase being reported yyyy-mmm-dd
# This part can be used again after the following data cleaning to double-check.
# print("Earliest Date: ", df.start_time.min())
# print("Latest Date:   ", df.end_time.max())

# ensure the data is within the date range
mask = (df['start_time'] > start_date) & (df['end_time'] < end_date)
df = df.loc[mask]

# remove overnight records
df = df.loc[df['start_time'].dt.date == df['end_time'].dt.date]

df.sort_values(by=['patient_id', 'start_time'], inplace=True)

# remove records beyond the normal working time
df = df.loc[df['start_time'].dt.time > pd.to_datetime('9:00:00').time()]
df = df.loc[df['end_time'].dt.time < pd.to_datetime('18:00:00').time()]

df.reset_index(drop=True, inplace=True)

# print(df)

'''
Create a new datafame to group each patient id.
Since staff appear in multiple days, the treatment should be different from the patient data.
For each staff member, the trace should be separated by dates.
'''

df['date'] = df['start_time'].dt.date  # Add a new column for grouping.
dfgrouped = df.groupby(['patient_id', 'date'], as_index=False).agg(
    {'start_time': ['min'], 'end_time': ['max'], 'x_location': ['first'], 'y_location': ['first']}
)

# Now the dfgrouped has two layers of column titles, and the following code is for flattening them.
flat_cols = []
for i in dfgrouped.columns:
    flat_cols.append(i[0]) # take the first element of the column heading only (ie ignore min, max, first)
dfgrouped.columns = flat_cols

dfgrouped['work_length'] = dfgrouped['end_time'] - dfgrouped['start_time']

dfgrouped = dfgrouped.dropna()
numberofdays = (dfgrouped['start_time'] - pd.to_datetime(start_date)).dt.days
dfgrouped['daynumber'] = numberofdays + 1 # adding one since counts from zero
dfgrouped['weeknumber'] = (numberofdays // 7) + 1 # adding one since counts from zero

# Time of Day
def ftod(x):
    if (x>12):
        tod = 'afternoon'
    else:
        tod = 'morning'
    return tod

dfgrouped['tod'] = dfgrouped.start_time.dt.hour.map(ftod)

# Work Length in Minutes
def get_seconds(time_delta):
    return time_delta.seconds

dfgrouped['work_length_minutes'] = dfgrouped['work_length'].apply(get_seconds)/60

'''
Grouped data cleaning
If there is any data being cleaned, please double check the input data.
'''
# print('\n--Grouped data cleaning (need to check the input data)--\n')
# print(dfgrouped.loc[(dfgrouped[['work_length_minutes']] != 0).all(axis=1)])
# print('The minimal work length is {} mins.'.format(dfgrouped['work_length_minutes'].min()))

'''
The following part is for describing and visualising the data
'''
# print('\n---Information about the input data---\n')
# print(df.columns)
# print("Earliest Date: ", df.start_time.min())
# print("Latest Date:   ", df.end_time.max())

# print('\n---Information about the grouped data---\n')
# print(dfgrouped.columns)
# print(dfgrouped.head())
# print(dfgrouped['work_length'].describe())

# print('\n---Time of Day Analysis---\n')
# print(dfgrouped.groupby('tod')['work_length'].mean(numeric_only=False))

'''
Export back to clean csv
'''
df.drop('date', axis=1, inplace=True)

# change time format before storing to be consistent with the database
# df['start_time'] = df['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
# df['end_time'] = df['end_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

# add two duplicate columns to be consistent with the database
df['start_time_ts'] = df['start_time']
df['end_time_ts'] = df['end_time']

# The original type of the following locations is object. They should be converted to float64.
df['x_location'] = pd.to_numeric(df['x_location'])
df['y_location'] = pd.to_numeric(df['y_location'])

'''
Please use vscode or notepad instead of excel to check the date time format.
Excel will automatically change the date format.
'''
# df.to_csv('tech/Input/{}_input.csv'.format(phase), index=False, date_format='%Y-%m-%d %H:%M:%S', encoding='utf-8')

# Change the order of columns to be consistent with the patient data.
dfgrouped = dfgrouped[['patient_id', 'start_time', 'end_time', 'work_length', 'date', 'tod', 'work_length_minutes']]

# dfgrouped.to_csv('tech/Grouped/{}_grouped_data.csv'.format(phase), index=False)

dfgrouped = df.groupby(['patient_id'], as_index=False)
print(dfgrouped.agg({'start_time': ['min'], 'end_time': ['max'], 'x_location': ['first'], 'y_location': ['first']}).iloc[:, 0])
