# %% [markdown]
# <a href="https://colab.research.google.com/github/djdunc/hercules/blob/main/data_import/heatmap.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data into a DataFrame
data = pd.read_csv("your_data.csv")

# Convert 'starttime' and 'endtime' to datetime objects
data['starttime'] = pd.to_datetime(data['starttime'])
data['endtime'] = pd.to_datetime(data['endtime'])

# Create a new DataFrame with minute intervals
start_time = data['starttime'].min()
end_time = data['endtime'].max()
index = pd.date_range(start=start_time.floor('T'), end=end_time.ceil('T'), freq='T')
agg_data = pd.DataFrame(index=index)

# Iterate through each row of your data and increment the count in the corresponding minute interval
for _, row in data.iterrows():
    start = row['starttime'].floor('T')
    end = row['endtime'].ceil('T')
    agg_data.loc[start:end] = agg_data.loc[start:end].fillna(0) + 1



# %%
# Create the heatmap using Seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(agg_data.T, cmap='viridis', annot=True, fmt='g')
plt.title('Aggregated Heatmap of Spatial Data (Minute Intervals)')
plt.xlabel('Time')
plt.ylabel('Counts')
plt.show()



