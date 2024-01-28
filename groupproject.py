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

##
#EDA and Visualizations
all_data.info()

all_data.describe()
all_data['Day_of_Week'] = ''
warnings.filterwarnings('ignore')

weekdays = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

all_data.columns = all_data.columns.str.replace(' ', '') #remove white space from all columns
weekday_cols = all_data.filter(like='weekday_is_').columns.tolist() #Days of the week columns

wday_dict = dict(zip(weekday_cols,weekdays))
# Loop through each row and find where the day of week column is flagged, use weekday dict to populate new factor column
for i in range(all_data.shape[0]):
  for key, value in wday_dict.items():
    if all_data[str(key)][i] == 1:
      day = value
      break
  all_data['Day_of_Week'][i] = day

print(all_data['Day_of_Week'].unique()) #Make sure this works!
new_all_data = all_data.drop(weekday_cols, axis=1) #Don't need these anymore
new_all_data.shape #Verify change in num cols

f, axs = plt.subplots(1, 2, figsize=(24, 8))
sns.kdeplot(data=new_all_data, x = new_all_data['shares']/10000, hue = 'Day_of_Week', multiple = 'stack', ax = axs[0], palette="husl")
plt.xlabel('Shares (in ten thousands)')
sns.kdeplot(data=new_all_data, x = np.log(new_all_data['shares']+1), hue = 'Day_of_Week', multiple = 'stack', ax = axs[1], palette="husl")
plt.xlabel('Shares')
plt.ylabel('Density')
plt.show()



#column_list = df.columns.tolist()
#print(column_list)


#df.corr()

#plt.figure(figsize=(16, 6))
#corr_heat = sns.heatmap(train.corr(numeric_only='True')[['loss']].sort_values(by='loss', ascending = False),vmin=-1, vmax=1, annot=True, cmap='BrBG');

#corr_heat.set_title('Continuous Features Correlating with Loss variable', fontdict={'fontsize':12}, pad=12)
#warnings.filterwarnings("ignore")
