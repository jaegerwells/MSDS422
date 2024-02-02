## Setting up Workbook to run 

import sys, os

import configparser
import subprocess
import warnings
import pprint

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


#for reproducibility

random.seed(540)

from config.definitions import ROOT_DIR
path_to_data = os.path.join(ROOT_DIR, 'data', 'OnlineNewsPopularity.csv')

all_data = pd.read_csv(path_to_data)
all_data['id'] = all_data.index + 1 

first_column = all_data.pop('id')
all_data.insert(0,'id', first_column)

print(all_data.head(5))


#EDA and Visualizations
all_data.info()
all_data.describe()
warnings.filterwarnings('ignore')
"""""
Data Cleaning, Begin with combining some columns
"""""
all_data['Day_of_Week'] = '' #Initialize blank columns in anticipation of combination effort
all_data['Channel'] = ''
weekdays = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
channels = ['Lifestyle','Entertainment','Bus','Socmed','Tech','World']

all_data.columns = all_data.columns.str.replace(' ', '') #remove white space from all columns
weekday_cols = all_data.filter(like='weekday_is_').columns.tolist() #Days of the week columns
channel_cols = all_data.filter(like='data_channel_is').columns.tolist() #Channel columns

wday_dict = dict(zip(weekday_cols,weekdays))
channel_dict = dict(zip(channel_cols,channels))

# Loop through each row and find where the day of week column is flagged, use weekday dict to populate new factor column
for i in range(all_data.shape[0]):
  for key, value in wday_dict.items():
    if all_data[str(key)][i] == 1:
      day = value
      break
  all_data['Day_of_Week'][i] = day
  #next do 
  for key, value in channel_dict.items():
    if all_data[str(key)][i] == 1:
      channel = value
      break
  all_data['Channel'][i] = channel

print(all_data['Day_of_Week'].unique()) #Make sure this works!
print(all_data['Channel'].unique()) #Make sure this works!
drop_list = weekday_cols + channel_cols
new_all_data = all_data.drop(drop_list, axis=1) #Don't need these anymore
#new_all_data = all_data.drop(channel_cols, axis=1) #Don't need these anymore
print(new_all_data.shape) #Verify change in num cols

f, axs = plt.subplots(2, 2, figsize=(24, 8))
sns.kdeplot(data=new_all_data, x = new_all_data['shares']/10000, hue = 'Day_of_Week', multiple = 'stack', ax = axs[0,0], palette="husl")
plt.xlabel('Shares (in ten thousands)')
sns.kdeplot(data=new_all_data, x = np.log(new_all_data['shares']+1), hue = 'Day_of_Week', multiple = 'stack', ax = axs[1,0], palette="husl")
axs[1,0].set_xlabel('log(Shares)')
plt.ylabel('Density')
sns.kdeplot(data=new_all_data, x = new_all_data['shares']/10000, hue = 'Channel', multiple = 'stack', ax = axs[0,1], palette="husl")
plt.xlabel('Shares (in ten thousands)')
sns.kdeplot(data=new_all_data, x = np.log(new_all_data['shares']+1), hue = 'Channel', multiple = 'stack', ax = axs[1,1], palette="husl")
plt.xlabel('log(Shares)')
plt.ylabel('Density')
plt.show(block=True) #set to false if you want the code to continue past the plot
"""""
This tells me there are lots of outliers, based on the raw shares. Confirming with boxplots below lol
"""""
sns.boxplot(data=new_all_data, x="shares", y="Day_of_Week")
plt.show()
sns.boxplot(data=new_all_data, x="shares", y="Channel")
plt.show()

#dropping the URL as it isn't useful information
column_to_drop = 'url'
new_all_data = new_all_data.drop(columns=[column_to_drop]) #[CW] I think we need to drop my str factor cols I just made before corr analysis
new_all_data = new_all_data.drop(['Day_of_Week','Channel'], axis=1)
new_all_data.shape

new_all_data.shares.describe()

explan_vars = new_all_data.drop(columns=['shares'])

target = new_all_data['shares']

explan_var_corr = -0.009

#iterating correlations to shares

for column in explan_vars.columns:
  correlation = target.corr(explan_vars[column])
  
  try:
    if correlation > explan_var_corr:
      print(f"Correlation between 'shares' and '{column}': {correlation}")
  except Exception as e:
    pass

    

#column_list = df.columns.tolist()
#print(column_list)





#plt.figure(figsize=(30, 15))
#corr_heat = sns.heatmap(new_all_data.corr(numeric_only='True')[['shares']].sort_values(by='shares', ascending = False), cmap='BrBG');

#corr_heat.set_title('Continuous Features Correlating with Share variable', fontdict={'fontsize':12}, pad=12)
#warnings.filterwarnings("ignore")

# Create a 'top_data_channel' column based on the data_channel columns
df['top_data_channel'] = df[['data_channel_is_lifestyle', 'data_channel_is_entertainment',
                             'data_channel_is_bus', 'data_channel_is_socmed',
                             'data_channel_is_tech', 'data_channel_is_world']].idxmax(axis=1)

# Print the unique values in the 'top_data_channel' column
unique_top_data_channels = df['top_data_channel'].unique()
print(f"Unique values in 'top_data_channel': {unique_top_data_channels}\n")

# Define a function to extract the last word from a string
def extract_last_word(channel):
    words = channel.split('_')
    return words[-1]

# Apply the function to the 'top_data_channel' column and create a new 'top_data_channel_last_word' column
df['top_data_channel_last_word'] = df['top_data_channel'].apply(extract_last_word)

# Print the unique values in the 'top_data_channel_last_word' column
unique_last_words = df['top_data_channel_last_word'].unique()
print(f"Unique last words in 'top_data_channel_last_word': {unique_last_words}\n")

import matplotlib.pyplot as plt
import seaborn as sns

# Box plot of 'shares' vs. 'top_data_channel_last_word'
plt.figure(figsize=(16, 8))
ax = sns.boxplot(x='top_data_channel_last_word', y='shares', data=df, palette='viridis')
plt.title('Correlation between Top Data Channel and Number of Shares')
plt.xlabel('Top Data Channel (Last Word)')
plt.ylabel('Number of Shares (scaled to 10,000)')

# Scale y-axis ticks and labels to 10,000
plt.yticks(ticks=[i*10000 for i in range(6)], labels=[f'{i*10000:.0f}' for i in range(6)])

# Create a secondary y-axis with the same scale
secondary_ax = ax.secondary_yaxis('right')
secondary_ax.set_ylabel('Number of Shares')

plt.show()



