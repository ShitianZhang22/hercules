# %% [markdown]
# <a href="https://colab.research.google.com/github/djdunc/hercules/blob/main/data_import/patient_viewer_seconds.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Seconds per step data
# 
# This script takes the P[n]_input.csv data and saves it with timing information between each step (in seconds) for use with the Processing visualisation sketch.

# %%
import pandas as pd
import datetime as dt

phase = 'P3' # edit which Phase you are analysing - this is used in graph and file generation
#patient = 'G1839'

df = pd.read_csv('{}_input.csv'.format(phase))
df # check what column headings in file

# df[df['Patient'] == patient].head() 

# %%
df['starttime'] = pd.to_datetime(df['starttime'], dayfirst=True) # datetime in YYYY-MM-DD HH:MM:SS format
df['endtime'] = pd.to_datetime(df['endtime'], dayfirst=True)

df['step_length_sec'] = (df['endtime'] - df['starttime']).astype('timedelta64[s]') 
#df[df['Patient'] == patient] 
df

# %% [markdown]
# Remove any steps that have a zero step_length time since we don't have a way to show multiple 0 readings.

# %%
df.drop(df.index[df['step_length_sec'] == 0], inplace = True)
df

# %%
df.step_length_sec.min()

# %%
# save all the data for one patient id to csv
#patient = 'G1839'
#journey = df[df['Patient'] == patient]
#print(journey.step_length_sec.sum())
#journey.to_csv('{}_journey.csv'.format(patient), index=False)

# %%
#journey

# %%
# remove any 0 value time steps then save
#patient = 'G1839'
#journey = df[df['Patient'] == patient]
#print(journey.step_length_sec.sum())
#journey.drop(journey.index[journey['step_length_sec'] == 0], inplace = True)
#journey.to_csv('{}_journey_no0.csv'.format(patient), index=False)

# %%
#journey

# %%
df.nlargest(10, 'xlocation')

# %%
df.ylocation.min()

# %%
df.nsmallest(10, 'ylocation')

# %%
df.nlargest(10, 'ylocation')

# %%
sorted_df = df.sort_values(by=['Patient', 'starttime'], ascending=True)
sorted_df.to_csv('{}_input_with_sec.csv'.format(phase), index=False)


