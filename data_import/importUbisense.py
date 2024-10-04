# %% [markdown]
# # Helper Script to Import Data from Ubisense
# 
# Script to import data from Ubisense "all rawdata" download. Reads in data, checks for any inconsistencies in tag recordings and outputs a cleaned data set for analysis. This is anticipated for use in preparing data for integration with observed data and environmental analysis.
# 
# Reads in a csv file in the following format:
# 
# `Patient, Location, from, to  `
# 
# It then cleans the data to:
# - split the location data in to x and y columns
# - rename the `from` `to` columns to avoid keyword conlicts 
# - deletes any rows that dont start with G or R
# - look for any records where start to finish are over 24 hours (since these are probably not valid or stale tags) and removes all records for that tag id
# - looks at each tag record and if any have a start and finish time that are more than an hour it deletes that individual entry (sometimes tag end times are a few hours after the end of a shift)
# - for each patient we look for any journeys greater than 6 hours and delete them (assumes incorrect tag allocation)
# - for each patient we look for any journeys less than 15 minutes and delete them 
# 
# Finally the output is saved to csv and has the format:
# 
# `Patient, Location, starttime,	endtime,	xlocation,	ylocation, step_length `

# %% [markdown]
# ## Initialisation
# 
# Import dependencies, create helper function, initialise some variables including loading in data. Files to import are based on the Ubisense "all rawdata" downloads from the reports section of their application interface. These are stored locally in the project "Teams" folder Data>ubisense>rawdata_smartspaces and usually have a filename such as "phase2_all_20220130.csv".

# %%
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import datetime as dt
import math
import seaborn as sns
import scipy.stats as sps
from scipy import stats

order_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
order_list_noweekend = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

pd.set_option('display.max_rows', 300) # specifies number of rows to show
pd.options.display.float_format = '{:40,.4f}'.format # specifies default number format to 4 decimal places
plt.style.use('ggplot') # specifies that graphs should use ggplot styling

##### Edit fields below before processing data #######

phase = 'P4_staff_september' # edit which Phase you are analysing - this is used in graph and file generation
# this script assumes CSV above has MM/DD/YYYY format - if not changes needed in next section below

start_date = '2022-7-6' # edit these for reducing processed download between 2 dates
end_date = '2022-10-01'

##### Edit fields above before processing data #######


df = pd.read_csv('{}_raw.csv'.format(phase))
df # check what column headings in file

# %% [markdown]
# ## Data Cleaning and Formatting
# 
# Data formatted for analysis. Note imported data from csv is in MM/DD/YYYY format - below we convert to a datetime in YYYY-MM-DD HH:MM:SS format. 
# 
# NEED TO MAKE SURE IF UBISENSE CHANGE OUTPUT OF THEIR RAW TABLE DOWNLOAD WE NEED TO RE ADD DAYFIRST FLAG BELOW.

# %%
df = df.dropna() # remove any rows with null values
df[['xlocation', 'ylocation']] = df['Location'].str.split(',', expand = True) # create seperate columns for the x y values
df.rename(columns={"from": "starttime", "to": "endtime"}, inplace=True) # renaming from and to column headings (from is a keyword)
df['starttime'] = pd.to_datetime(df['starttime']) #, dayfirst=True)
df['endtime'] = pd.to_datetime(df['endtime']) #, dayfirst=True)
df

# %%
#df.rename(columns={'Technician':'Patient'}, inplace=True)
df[~df["Patient"].str.startswith(('G', 'R', 'C', 'S'))] # look for any records that dont (~) start with G or R 

# %%
df = df.drop(df[~df["Patient"].str.startswith(('G', 'R', 'C', 'S'))].index) # and then drop those rows
df

# %%
df['step_length'] = df['endtime'] - df['starttime'] # add in variable that reports the time at each step between records
print(df[df['step_length'] > pd.Timedelta(2, 'h')]) # list out any where the time iterval is greater than 2 hours for an individual step

# %%
print(df[df['step_length'] < pd.Timedelta(0, 'h')]) # list out any where the time iterval is greater than 2 hours for an individual step

# %%
df[df['Patient'] == 'R0651'] # If you want to check data before deletion, use this function to check to see the record that seems to be too long

# %%
df = df.drop(df[df['step_length'] > pd.Timedelta(2, 'h')].index) # remove any individual step records that are more than 2 hours
df = df.drop(df[df['step_length'] < pd.Timedelta(0, 'h')].index) # remove any individual step records that are more than 2 hours

# %%
# check the start and end dates of the phase being reported yyyy-mmm-dd
print("Earliest Date: ", df.starttime.min())
print("Latest Date:   ", df.endtime.max())
#df.dtypes

# %%
df.sort_values(by=['starttime'])
df

# %%
mask = (df['starttime'] > start_date) & (df['starttime'] <= end_date)
df.sort_values(by=['starttime'])
df = df.loc[mask]
df

# %%
# check the start and end dates of the phase being reported yyyy-mmm-dd
print("Earliest Date: ", df.starttime.min())
print("Latest Date:   ", df.endtime.max())

# %%
df.nsmallest(10, 'step_length')

# %%
df.sort_values(by=['starttime'])

# %%
df.dtypes # list out the updated column names and data types

# %% [markdown]
# ## Create a new dataframe to group each patient id
# 
# The code belows gets the earliest and latest times from a patient allowing the total journey time to be tracked. 

# %%
dfgrouped = df.groupby('Patient', as_index = False).agg(
    {'starttime': ['min'], 'endtime': ['max'], 'xlocation': ['first'], 'ylocation': ['first']})
dfgrouped

# %% [markdown]
# The code below changes column headers and continues to format and clean the data for subsequent analysis.

# %%
flat_cols = []
for i in dfgrouped.columns:
    flat_cols.append(i[0]) # take the first element of the column heading only (ie ignore min, max, first)
dfgrouped.columns = flat_cols

# %% [markdown]
# Add in length of journey to each grouped patient id

# %%
dfgrouped['visit_length'] = dfgrouped['endtime'] - dfgrouped['starttime']

# %%
dfgrouped

# %% [markdown]
# # Delete Outlier Patient Journeys
# Check to see if there are any groups of patient records that are over 1 day in length and then remove them from data frame. Assumption here is that if tag timings are greater than 24 hours it is a tag that has been aborted, lost or forgotton about. NOTE: this will remove any records where the same Patient ID has been used twice. 

# %%
print(dfgrouped[dfgrouped['visit_length'] > pd.Timedelta(1, 'd')]) # list out the ones to be deleted

# %%
dfgrouped = dfgrouped.drop(dfgrouped[dfgrouped['visit_length'] > pd.Timedelta(1, 'd')].index) # delete those over 1 day

# %%
dfgrouped['visit_length'].max() # check to see what max length of visit is after first cull

# %% [markdown]
# Check to see if there are any tags that have been left unchecked at end of day - and delete those records from main dataframe. NOTE: could try and delete the readings that are overnight but difficult pattern to identify - since low percentage of results, just delete - think these are tags that are not checked back in properly?

# %%
dfgrouped[dfgrouped['visit_length'] > pd.Timedelta(6, 'h')] # check to see how many patient journeys are over 6 hours - probably not valid so delete

# %%
dfgrouped = dfgrouped.drop(dfgrouped[dfgrouped['endtime'] - dfgrouped['starttime'] > pd.Timedelta(6, 'h')].index) # delete those over 5 hours

# %% [markdown]
# Check to see how many patient journeys are implausibly short - ie under 15 minutes - and delete those.

# %%
dfgrouped['visit_length'].min() # check to see what min length of visit is after first cull

# %%
dfgrouped[dfgrouped['visit_length'] < pd.Timedelta(15, 'm')] # check to see how many patient journeys are under 15 mins - probably not valid so delete

# %%
dfgrouped = dfgrouped.drop(dfgrouped[dfgrouped['endtime'] - dfgrouped['starttime'] < pd.Timedelta(15, 'm')].index) # delete those under 15 min

# %%
dfgrouped

# %%
# make a copy of records (df3) from df that have an equivalent Patient id in dfgrouped
df3 = df[df['Patient'].isin(dfgrouped['Patient'])]  
print('start record count: ', df.Patient.count())
print('clean record count: ', df3.Patient.count())

# %% [markdown]
# # Export back to clean csv
# Assumes user will then move to appropriate "clean data" folder so just using generic output.csv filename.

# %%
df3.to_csv('{}_input.csv'.format(phase), index=False) # note use subset of records in df3

# %% [markdown]
# To end, print out some patient journey stats for info!

# %%
dfgrouped['visit_length'].describe()

# %% [markdown]
# End


