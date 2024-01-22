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


#for reproducibility

random.seed(540)

from config.definitions import ROOT_DIR
path_to_data = os.path.join(ROOT_DIR, 'data', 'OnlineNewsPopularity.csv')

df = pd.read_csv(path_to_data)

df['id'] = df.index + 1 

first_column = df.pop('id')
df.insert(0,'id', first_column)

print(df.head(5))


#EDA and Visualizations




column_list = df.columns.tolist()
print(column_list)


df.info()

df.describe()



#df.corr()

#plt.figure(figsize=(16, 6))
#corr_heat = sns.heatmap(train.corr(numeric_only='True')[['loss']].sort_values(by='loss', ascending = False),vmin=-1, vmax=1, annot=True, cmap='BrBG');

#corr_heat.set_title('Continuous Features Correlating with Loss variable', fontdict={'fontsize':12}, pad=12)
#warnings.filterwarnings("ignore")
