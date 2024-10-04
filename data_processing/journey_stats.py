"""
Box Plots for journey times through BX
# Import data, group by patient id etc .....
"""

'''
Initialisation
Import dependencies, initialise some variables including loading in data. Files to import are based on filtered 
Ubisense "all rawdata" downloads. These are stored in the github folder "data" of this repo.
'''

from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import datetime as dt
import math
import seaborn as sns
import scipy.stats as sps
from scipy import stats
from google.colab import files

order_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
order_list_noweekend = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

pd.set_option('display.max_rows', 300)  # specifies number of rows to show
pd.options.display.float_format = '{:40,.4f}'.format  # specifies default number format to 4 decimal places
plt.style.use('ggplot')  # specifies that graphs should use ggplot styling
fig_size = (16,10)  # how big the exported figures should be - width, height - in inches!
ymax_boxplot = 240  # max scale on the box plots to normalise across phases

# %% [markdown]
# ## Edit this section per run of data processing

# %%
# ------------------
# Edit these lines per run of data processing
# ------------------
#phase = 'P1' # edit which Phase you are analysing - this is used in graph and file generation
#p_start_date = dt.date(2021, 10, 11) # YYYY, MM, DD
#p_end_date = dt.date(2021, 11, 12)
#phase = 'P2' # edit which Phase you are analysing - this is used in graph and file generation
#p_start_date = dt.date(2021, 11, 30) # YYYY, MM, DD
#p_end_date = dt.date(2022, 1, 28)
#phase = 'P3' # edit which Phase you are analysing - this is used in graph and file generation
#p_start_date = dt.date(2022, 2, 23) # YYYY, MM, DD
#p_end_date = dt.date(2022, 5, 6)
phase = 'P4' # edit which Phase you are analysing - this is used in graph and file generation
p_start_date = dt.date(2022, 9, 7) # YYYY, MM, DD
p_end_date = dt.date(2023, 3, 1)
# ------------------

# read in CSV data
df = pd.read_csv('{}_input.csv'.format(phase))
# convert dates into ones that can be used in pandas
df['starttime'] = pd.to_datetime(df['starttime']) # datetime in YYYY-MM-DD HH:MM:SS format
df['endtime'] = pd.to_datetime(df['endtime'])

df.sort_values(by='starttime', inplace = True)
df # check what column headings in file

# %%
# check the start and end dates of the phase being reported yyyy-mmm-dd
print("Earliest Date: ", df.starttime.min())
print("Latest Date:   ", df.endtime.max())

# %%
print(df[df['starttime'] < pd.to_datetime(p_start_date)])
print(df[df['endtime'] > pd.to_datetime(p_end_date)])
df = df.drop(df[df['starttime'] < pd.to_datetime(p_start_date)].index)
df = df.drop(df[df['endtime'] > pd.to_datetime(p_end_date)].index)


# %%
print(df[df['starttime'] < pd.to_datetime(p_start_date)])
print(df[df['endtime'] > pd.to_datetime(p_end_date)])

# %% [markdown]
# # Create a new dataframe to group each patient id - df_final
# 
# The code belows gets the earliest and latest times from a patient allowing the total journey time to be tracked.

# %%
dfgrouped = df.groupby('Patient', as_index = False).agg(
    {'starttime': ['min'], 'endtime': ['max'], 'xlocation': ['first'], 'ylocation': ['first']})
dfgrouped.head()

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
print(dfgrouped[dfgrouped['visit_length'] > pd.Timedelta(1, 'd')]) # this should return no results!

# %%
dfgrouped['visit_length'].describe()

# %% [markdown]
# Add in a day of week column value.

# %%
dfgrouped['dayofweek'] = dfgrouped['starttime'].dt.day_name()
df_final= dfgrouped.dropna()
df_final

# %%
numberofdays = (df_final['starttime'] - pd.to_datetime(p_start_date)).dt.days
df_final['daynumber'] = numberofdays + 1 # adding one since counts from zero
df_final['weeknumber'] = (numberofdays // 7) + 1 # adding one since counts from zero
df_final

# %% [markdown]
# # Analysis - df_final
# 
# Explore the "df_final" grouped data and make some intital plots.

# %% [markdown]
# ## Day of the Week Analysis

# %%
dayoftheweekmean = df_final.groupby('dayofweek')['visit_length'].mean(numeric_only=False)
dayoftheweekmean = dayoftheweekmean.reindex(index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
dayoftheweekmean

# %% [markdown]
# ## Week of Phase Analysis

# %%
weekofphasemean = df_final.groupby('weeknumber')['visit_length'].mean(numeric_only=False)
weekofphasemean

# %% [markdown]
# ## Day of Phase Analysis

# %%
dayofphasemean = df_final.groupby('daynumber')['visit_length'].mean(numeric_only=False)
dayofphasemean

# %% [markdown]
# 
# ## Time of Day Analysis
# 
# Afternoon is defined as a journey starting from 13:00 onwards whereas any journey starting prior to 13:00 is defined as starting in the morning.

# %%
def ftod(x):
    if (x>12):
        tod = 'afternoon'
    else:
        tod = 'morning'
    return tod

df_final['tod'] = df_final.starttime.dt.hour.map(ftod)

timeofdaymean = df_final.groupby('tod')['visit_length'].mean(numeric_only=False)
timeofdaymean # print out mean time for am pm

# %% [markdown]
# ## Day of the Week Analysis

# %%
df_dayoftheweekmean = pd.DataFrame(data=dayoftheweekmean)
df_dayoftheweekmean

# %%
Monday_count = df_final['dayofweek'][df_final['dayofweek']=='Monday'].count()
Tuesday_count = df_final['dayofweek'][df_final['dayofweek']=='Tuesday'].count()
Wednesday_count = df_final['dayofweek'][df_final['dayofweek']=='Wednesday'].count()
Thursday_count = df_final['dayofweek'][df_final['dayofweek']=='Thursday'].count()
Friday_count = df_final['dayofweek'][df_final['dayofweek']=='Friday'].count()
Saturday_count = df_final['dayofweek'][df_final['dayofweek']=='Saturday'].count()
Sunday_count = df_final['dayofweek'][df_final['dayofweek']=='Sunday'].count()

weekday_data = {'dayofweek':['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
          'count': [Monday_count , Tuesday_count , Wednesday_count , Thursday_count , Friday_count , Saturday_count , Sunday_count ]}

weekday_df = pd.DataFrame(weekday_data)
weekday_df

# %% [markdown]
# ## Start of Starting Hour Analysis
# 
# Here we look to analyse if there are any differences by the hour at which the patient journey starts.

# %%
df_final['hour'] = df_final['starttime'].dt.hour
df_final

# %%
hourlymean = df_final.groupby('hour')['visit_length'].mean(numeric_only=False)
hourlymean

# %%
df_final.dtypes

# %%
nine_count = df_final['hour'][df_final['hour']==9].count()
ten_count = df_final['hour'][df_final['hour']==10].count()
eleven_count = df_final['hour'][df_final['hour']==11].count()
twelve_count = df_final['hour'][df_final['hour']==12].count()
one_count = df_final['hour'][df_final['hour']==13].count()
two_count = df_final['hour'][df_final['hour']==14].count()
three_count = df_final['hour'][df_final['hour']==15].count()
four_count = df_final['hour'][df_final['hour']==16].count()

print('09:00 ', nine_count)
print('10:00 ', ten_count)
print('11:00 ', eleven_count)
print('12:00 ', twelve_count)
print('13:00 ', one_count)
print('14:00 ', two_count)
print('15:00 ', three_count)
print('16:00 ', four_count)

print('total:', nine_count+ten_count+eleven_count+twelve_count+one_count+two_count+three_count+four_count,
      '/', df_final.Patient.count())

# %% [markdown]
# ## Patient Condition Analysis
# 
# This analysis begins to compare the journeys of patients with either Glaucoma or Medical Retina.

# %%
df_final['condition'] = df_final['Patient'].str[0]
conditionmean = df_final.groupby('condition')['visit_length'].mean(numeric_only=False)
conditionmean

# %%
glaucoma_count = df_final['condition'][df_final['condition']=='G'].count()
retina_count = df_final['condition'][df_final['condition']=='R'].count()
cataract_count = df_final['condition'][df_final['condition']=='C'].count()
print('Medical Retinal', retina_count)
print('Glaucoma', glaucoma_count)
print('Cataract', cataract_count)
print(retina_count + glaucoma_count + cataract_count)

# %% [markdown]
# ## Converting Visit Length to Minutes
# 
# The following code converts the visit length into a more usable data format. The final data is an integer representing the number of minutes that the journey took to complete.

# %%
def get_seconds(time_delta):
    return time_delta.seconds

time_delta_series = df_final['visit_length']

df_final['visit_length_minutes'] = time_delta_series.apply(get_seconds)/60
df_final

# %% [markdown]
# ## Initial Plots
# 
# These are the starting visualisations. Final versions are seen later in the notebook.

# %%
axdayoftheweek = sns.boxplot(x="dayofweek", y="visit_length_minutes", order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] , data=df_final)

# %%
axtod = sns.boxplot(x="tod", y="visit_length_minutes", data=df_final)

# %%
axhour = sns.boxplot(x="hour", y="visit_length_minutes", data=df_final)

# %%
axcondition = sns.boxplot(x="condition", y="visit_length_minutes", data=df_final)

# %%
axweeknumber = sns.boxplot(x="weeknumber", y="visit_length_minutes", data=df_final)

# %%
axdaynumber = sns.boxplot(x="daynumber", y="visit_length_minutes", data=df_final)

# %% [markdown]
# # Outlier Removal - df_outliers
# 
# There are many odd outliers in this data. This section removes these outliers.

# %%
df_outliers = df_final[(np.abs(stats.zscore(df_final['visit_length_minutes'])) < 3)] # finds any outliers 3 SD's away from mean
df_outliers.loc[(df_outliers[['visit_length_minutes']] < 10).all(axis=1)]
print(df_outliers.loc[(df_outliers[['visit_length_minutes']] < 10).all(axis=1)].count())

# %% [markdown]
# ## Removal of 0 Minute Journeys
# 
# The section below creates a dataframe where journeys that have a journey time of 0 minutes are removed.

# %%
df_nozero = df_outliers.loc[(df_outliers[['visit_length_minutes']] != 0).all(axis=1)]
df_nozero

# %% [markdown]
# # Start of Final Analysis - df_outlier

# %%
mean = df_outliers['visit_length_minutes'].mean()
print(df_outliers['visit_length_minutes'].mean())
print(df_nozero['visit_length_minutes'].mean())

# %% [markdown]
# ## Plot all journeys as histogram
# 
# Show distribution of patients as histogram and KDE (kernel denisty estimation)

# %%
ax = sns.displot(df_outliers['visit_length_minutes'], kde=True, alpha=.4, rug=True)
plt.title('{} Visit Length Distribution Plot'.format(phase))
plt.show()
ax.savefig('{}_visitlength_histogram.png'.format(phase), bbox_inches='tight')
files.download('{}_visitlength_histogram.png'.format(phase))

# %%
ax = sns.displot(data=df_outliers, x=df_outliers['visit_length_minutes'], hue=df_outliers['condition'], kind="kde")
plt.title('{} Visit Length Distribution Plot'.format(phase))
plt.show()
ax.savefig('{}_visitlength_KDEplot.png'.format(phase), bbox_inches='tight')
files.download('{}_visitlength_KDEplot.png'.format(phase))

# %%
from sklearn.linear_model import LinearRegression
# Creating a Linear Regression model on our data
lin = LinearRegression()
lin.fit(df_outliers[['daynumber']], df_outliers['visit_length_minutes'])
# Creating a plot
ax = df_outliers.plot.scatter(x='daynumber', y='visit_length_minutes', alpha=.5)
ax.plot(df_outliers['daynumber'], lin.predict(df_outliers[['daynumber']]), c='r')
plt.ylim(0, ymax_boxplot)
plt.xlabel('Days into trial')
plt.ylabel('Visit Length in Minutes')
plt.title('{} Visit Length Change Over Trial - Slope={} Intercept={:.0f}'.format(phase, lin.coef_, lin.intercept_))
plt.show()

figure1 = ax.get_figure()
figure1.savefig('{}_visitlength_regresion.png'.format(phase), bbox_inches='tight')
files.download('{}_visitlength_regresion.png'.format(phase))

# %%
#To retrieve the intercept:
print(lin.intercept_)

#For retrieving the slope:
print(lin.coef_)

# %% [markdown]
# ## Day of Week Analysis and Plots

# %%
sns.set_style('whitegrid')

sns.set_theme(palette="colorblind")
f,ax = plt.subplots(figsize=(fig_size))
svm = sns.boxplot(x="dayofweek", y="visit_length_minutes", data=df_outliers, order = order_list)

# Calculate number of obs per group & median to position labels
medians = df_outliers.groupby(['dayofweek'])['visit_length_minutes'].median().values
nobs = df_outliers['dayofweek'].value_counts(sort=False).values
nobs = [str(x) for x in nobs.tolist()]
nobs = ["n: " + i for i in nobs]

pos = range(len(nobs))
for tick,label in zip(pos,svm.get_xticklabels()):
    svm.text(pos[tick],
            medians[tick] + 1,
            nobs[tick],
            horizontalalignment='center',
            size='x-small',
            color='w',
            weight='semibold')


plt.ylim(0, ymax_boxplot)

svm.set_title('{} Visit Length in Minutes by Day of the Week'.format(phase))
svm.set_ylabel('Visit Length in Minutes')
svm.set_xlabel('Day of the Week')
svm.axhline(mean, linestyle = '--', color = 'black', label = 'Overall Mean')
svm.legend(bbox_to_anchor = (0.85, 1), loc = 'upper center')
figure1 = svm.get_figure()
figure1.savefig('{}_visitlength_7dayweek.png'.format(phase), bbox_inches='tight')
files.download('{}_visitlength_7dayweek.png'.format(phase))

# %%
sns.set_style('whitegrid')
sns.set_theme(palette="colorblind")
f,ax = plt.subplots(figsize=(fig_size))
svm = sns.boxplot(x="dayofweek", y="visit_length_minutes", data=df_outliers, order = order_list_noweekend)

# Calculate number of obs per group & median to position labels
medians = df_outliers.groupby(['dayofweek'])['visit_length_minutes'].median().values
nobs = df_outliers['dayofweek'].value_counts(sort=False).values
nobs = [str(x) for x in nobs.tolist()]
nobs = ["n: " + i for i in nobs]

pos = range(len(nobs))
for tick,label in zip(pos,svm.get_xticklabels()):
    svm.text(pos[tick],
            medians[tick] + 1,
            nobs[tick],
            horizontalalignment='center',
            size='x-small',
            color='w',
            weight='semibold')

plt.ylim(0, ymax_boxplot)

svm.set_title('{} Visit Length in Minutes by Day of the Week'.format(phase))
svm.set_ylabel('Visit Length in Minutes')
svm.set_xlabel('Day of the Week')
svm.axhline(mean, linestyle = '--', color = 'black', label = 'Overall Mean')
svm.legend(bbox_to_anchor = (0.85, 1), loc = 'upper center')
figure1 = svm.get_figure()
figure1.savefig('{}_visitlength_5dayweek.png'.format(phase), bbox_inches='tight')
files.download('{}_visitlength_5dayweek.png'.format(phase))

# %% [markdown]
# ## Time of Day Analysis and Plots

# %%
sns.set_style('whitegrid')
sns.set_theme(palette="colorblind")
f,ax = plt.subplots(figsize=(fig_size))
svm = sns.boxplot(x="tod", y="visit_length_minutes", data=df_outliers)

# Calculate number of obs per group & median to position labels
medians = df_outliers.groupby(['tod'])['visit_length_minutes'].median().values
nobs = df_outliers['tod'].value_counts(sort=False).values
nobs = [str(x) for x in nobs.tolist()]
nobs = ["n: " + i for i in nobs]

pos = range(len(nobs))
for tick,label in zip(pos,svm.get_xticklabels()):
    svm.text(pos[tick],
            medians[tick] + 1,
            nobs[tick],
            horizontalalignment='center',
            size='x-small',
            color='w',
            weight='semibold')

plt.ylim(0, ymax_boxplot)

svm.set_title('{} Visit Length in Minutes (PM after 13:00)'.format(phase))
svm.set_ylabel('Visit Length in Minutes')
svm.set_xlabel('Time of Day')
svm.axhline(mean, linestyle = '--', color = 'black', label = 'Overall Mean')
svm.legend(bbox_to_anchor = (0.85, 1), loc = 'upper center')
figure1 = svm.get_figure()
figure1.savefig('{}_visitlength_am_pm.png'.format(phase), bbox_inches='tight')
files.download('{}_visitlength_am_pm.png'.format(phase))

# %% [markdown]
# ## Condition Analysis and Plots

# %%
sns.set_style('whitegrid')
sns.set_theme(palette="colorblind")
f,ax = plt.subplots(figsize=(fig_size))
svm = sns.boxplot(x="condition", y="visit_length_minutes", data=df_outliers)

# Calculate number of obs per group & median to position labels
medians = df_outliers.groupby(['condition'])['visit_length_minutes'].median().values
nobs = df_outliers['condition'].value_counts(sort=False).values
nobs = [str(x) for x in nobs.tolist()]
nobs = ["n: " + i for i in nobs]

pos = range(len(nobs))
for tick,label in zip(pos,svm.get_xticklabels()):
    svm.text(pos[tick],
            medians[tick] + 1,
            nobs[tick],
            horizontalalignment='center',
            size='x-small',
            color='w',
            weight='semibold')

plt.ylim(0, ymax_boxplot)

svm.set_title('{} Visit Length in Minutes by Condition'.format(phase))
svm.set_ylabel('Visit Length in Minutes')
svm.set_xlabel('Time of Day')
svm.axhline(mean, linestyle = '--', color = 'black', label = 'Overall Mean')
svm.legend(bbox_to_anchor = (0.85, 1), loc = 'upper center')
figure1 = svm.get_figure()
figure1.savefig('{}_visitlength_G_R.png'.format(phase), bbox_inches='tight')
files.download('{}_visitlength_G_R.png'.format(phase))

# %%
#axcondition = sns.boxplot(x="condition", y="visit_length_minutes", data=df_outliers)

# %% [markdown]
# ## Starting Hour Analysis and Plots

# %%
#axhour = sns.boxplot(x="hour", y="visit_length_minutes", data=df_outliers)

# %%
sns.set_style('whitegrid')
sns.set_theme(palette="colorblind")
f,ax = plt.subplots(figsize=(fig_size))
svm = sns.boxplot(x="hour", y="visit_length_minutes", data=df_outliers)

# Calculate number of obs per group & median to position labels
medians = df_outliers.groupby(['hour'])['visit_length_minutes'].median().values
nobs = df_outliers['hour'].value_counts(sort=False).values
nobs = [str(x) for x in nobs.tolist()]
nobs = ["n: " + i for i in nobs]

pos = range(len(nobs))
for tick,label in zip(pos,svm.get_xticklabels()):
    svm.text(pos[tick],
            medians[tick] + 1,
            nobs[tick],
            horizontalalignment='center',
            size='x-small',
            color='w',
            weight='semibold')

plt.ylim(0, ymax_boxplot)

svm.set_title('{} Visit Length in Minutes by Starting Hour'.format(phase))
svm.set_ylabel('Visit Length in Minutes')
svm.set_xlabel('Starting Hour')
svm.axhline(mean, linestyle = '--', color = 'black', label = 'Overall Mean')
svm.legend(bbox_to_anchor = (0.85, 1), loc = 'upper center')
figure1 = svm.get_figure()
figure1.savefig('{}_visitlength_entryhour.png'.format(phase), bbox_inches='tight')
files.download('{}_visitlength_entryhour.png'.format(phase))

# %% [markdown]
# ## Week Numbers

# %%
sns.set_style('whitegrid')
sns.set_theme(palette="colorblind")
f,ax = plt.subplots(figsize=(fig_size))
svm = sns.boxplot(x="weeknumber", y="visit_length_minutes", data=df_outliers)

# Calculate number of obs per group & median to position labels
medians = df_outliers.groupby(['weeknumber'])['visit_length_minutes'].median().values
nobs = df_outliers['weeknumber'].value_counts(sort=False).values
nobs = [str(x) for x in nobs.tolist()]
nobs = ["n: " + i for i in nobs]

pos = range(len(nobs))
for tick,label in zip(pos,svm.get_xticklabels()):
    svm.text(pos[tick],
            medians[tick] + 1,
            nobs[tick],
            horizontalalignment='center',
            size='x-small',
            color='w',
            weight='semibold',
            rotation=90)

plt.ylim(0, ymax_boxplot)

svm.set_title('{} Visit Length in Minutes by Week Number'.format(phase))
svm.set_ylabel('Visit Length in Minutes')
svm.set_xlabel('Week Number')
svm.axhline(mean, linestyle = '--', color = 'black', label = 'Overall Mean')
svm.legend(bbox_to_anchor = (0.85, 1), loc = 'upper center')
figure1 = svm.get_figure()
figure1.savefig('{}_visitlength_week_number.png'.format(phase), bbox_inches='tight')
files.download('{}_visitlength_week_number.png'.format(phase))

# %% [markdown]
# ## Day Numbers

# %%
sns.set_style('whitegrid')
sns.set_theme(palette="colorblind")
f,ax = plt.subplots(figsize=(fig_size))
svm = sns.boxplot(x="daynumber", y="visit_length_minutes", data=df_outliers)
svm.xaxis.set_major_locator(ticker.MultipleLocator(7))
svm.xaxis.set_major_formatter(ticker.ScalarFormatter())

# Calculate number of obs per group & median to position labels
medians = df_outliers.groupby(['daynumber'])['visit_length_minutes'].median().values
nobs = df_outliers['daynumber'].value_counts(sort=False).values
nobs = [str(x) for x in nobs.tolist()]
nobs = ["n: " + i for i in nobs]

pos = range(len(nobs))
for tick,label in zip(pos,svm.get_xticklabels()):
    svm.text(pos[tick],
            medians[tick] + 1,
            nobs[tick],
            horizontalalignment='center',
            size='x-small',
            color='w',
            weight='semibold',
            rotation=90)

plt.ylim(0, ymax_boxplot)

svm.set_title('{} Visit Length in Minutes by Day Number'.format(phase))
svm.set_ylabel('Visit Length in Minutes')
svm.set_xlabel('Day Number')
svm.axhline(mean, linestyle = '--', color = 'black', label = 'Overall Mean')
svm.legend(bbox_to_anchor = (0.85, 1), loc = 'upper center')
figure1 = svm.get_figure()
figure1.savefig('{}_visitlength_day_number.png'.format(phase), bbox_inches='tight')
files.download('{}_visitlength_day_number.png'.format(phase))

# %% [markdown]
# # Summary Statistics of Each Analysis
# 
# The section below gets the summary statistics of each analysis done. These are used in the PowerPoint in the Findings folder in the MS Teams.

# %%
dayoftheweekmean = df_outliers.groupby('dayofweek')['visit_length'].mean(numeric_only=False)
dayoftheweekmean = dayoftheweekmean.reindex(index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# %%
df_outliers['visit_length_minutes'].describe()

# %%
df_final['visit_length_minutes'].describe()

# %%
df_final['visit_length_minutes'].median()

# %%
df_outliers['visit_length'].describe()

# %%
df_final['visit_length'].describe()

# %%
sns.set_style('whitegrid')
sns.set_theme(palette="colorblind")
f,ax = plt.subplots(figsize=(fig_size))
svm = sns.boxplot(y="visit_length_minutes", data=df_outliers)

# Calculate number of obs
nobs = df_outliers['Patient'].count()

plt.ylim(0, ymax_boxplot)

svm.set_title('Average {} Visit Length.'.format(phase))
svm.set_ylabel('Visit Length in Minutes')
svm.set_xlabel('Total (n= {})'.format(nobs))
svm.axhline(mean, linestyle = '--', color = 'black', label = 'Overall Mean')
svm.legend(bbox_to_anchor = (0.85, 1), loc = 'upper center')
figure1 = svm.get_figure()
figure1.savefig('{}_visitlength.png'.format(phase), bbox_inches='tight')
files.download('{}_visitlength.png'.format(phase))

# %%
df_outliers.nsmallest(5, 'visit_length')

# %%
df_outliers.nsmallest(5, 'visit_length_minutes')

# %%
df_outliers.nlargest(5, 'visit_length_minutes')

# %%
df_outliers.nlargest(5, 'visit_length')

# %%
df_final.nlargest(20, 'visit_length')

# %%
df_outliers.nlargest(20, 'visit_length')

# %% [markdown]
# # Save results

# %% [markdown]
# ## CSV Creation
# 
# This creates a CSV of the patient journies which can be used for subsequent analysis (e.g. cluster analysis).

# %%
phase2df = df_outliers[['Patient', 'starttime', 'endtime', 'visit_length', 'dayofweek', 'tod', 'hour', 'condition', 'visit_length_minutes']].copy()
phase2df.to_csv('{}_grouped_data.csv'.format(phase), index=False)
files.download('{}_grouped_data.csv'.format(phase))


