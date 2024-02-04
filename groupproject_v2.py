## Setting up Workbook to run 

import sys, os

import configparser
import subprocess
import warnings
import pprint

import math
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


#for reproducibility

random.seed(540)

#Loading the Dataset
from config.definitions import ROOT_DIR
path_to_data = os.path.join(ROOT_DIR, 'data', 'OnlineNewsPopularity.csv')


df = pd.read_csv(path_to_data)
df['id'] = df.index + 1 

first_column = df.pop('id')
df.insert(0,'id', first_column)

print(df.head(3))

#EDA and Visualizations
df.info()

df.describe()
warnings.filterwarnings('ignore')

df.hist(figsize=(20,20))
plt.show()

df.columns=df.columns.str.replace(" ","")

df = df.drop('url',axis=1)

cor=df.corr()
#ns.heatmap(cor)
plt.figure(figsize=(15,15))
df_lt = cor.where(np.tril(np.ones(cor.shape)).astype(bool))
sns.heatmap(df_lt,cmap='BrBG')

num_of_nowords=df[df['n_tokens_content']==0].index
print('number of news with no words',num_of_nowords.size)

#shortening weekday and channel column header titles
df.columns = df.columns.str.replace('weekday_is_', '')
df.columns = df.columns.str.replace('data_channel_is_', '')

df.info()

df = df[df['n_tokens_content'] != 0]

#dropping columns that are not valuable for the analysis
df = df.drop('timedelta',axis=1)
df= df.drop(["n_non_stop_unique_tokens","n_non_stop_words","kw_avg_min"],axis=1)

df['shares'].describe()

#determine the appropriate threshold for number of shares for feature engineering.
df['shares'].median()

df['popularity'] = df['shares'].apply(lambda x: 0 if x <1400 else 1)

plt.figure(figsize=(10,5))
ax = sns.scatterplot(y='shares', x='n_tokens_content', data=df)

a,b = df['shares'].mean(),df['shares'].median()

weekday = df.columns.values[26:33]
weekday

#Visual of popular vs. unpopular news across the week
Unpop=df[df['shares']<b]
Pop=df[df['shares']>=b]
Unpop_day = Unpop[weekday].sum().values
Pop_day = Pop[weekday].sum().values

fig = plt.figure(figsize = (13,5))
plt.title("Count of popular vs unpopular news over different days of the week", fontsize = 16)

plt.bar(np.arange(len(weekday)),Pop_day,width=0.3,align='center',color='b',label='Popular')
plt.bar(np.arange(len(weekday))-0.3,Unpop_day,width=0.3,align='center',color='y',label='Unpopular')

plt.xticks(np.arange(len(weekday)),weekday)
plt.ylabel('Count',fontsize=15)
plt.xlabel('Days of the Week',fontsize=17)


plt.legend(loc = 'upper right')
plt.tight_layout()
plt.show()

channel=df.columns[9:15]
channel

#count of popular vs. unpopular news over different channels
Unpop2=df[df['shares']<b]
Pop2=df[df['shares']>=b]
Unpop_day2 = Unpop2[channel].sum().values
Pop_day2 = Pop2[channel].sum().values
fig = plt.figure(figsize = (13,5))
plt.title("Count of popular vs unpopular news over different channels", fontsize = 16)
plt.bar(np.arange(len(channel)), Pop_day2, width = 0.3, align="center", color = 'r', \
          label = "popular")
plt.bar(np.arange(len(channel)) - 0.3, Unpop_day2, width = 0.3, align = "center", color = 'g', \
          label = "unpopular")
plt.xticks(np.arange(len(channel)),channel)
plt.ylabel("Count", fontsize = 12)
plt.xlabel("Channel", fontsize = 12)
    
plt.legend(loc = 'upper right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(df.shares,color='g')
plt.xlabel('Records')
plt.ylabel('No of Shares')
plt.title('Shares Plot')
plt.show()

plt.figure(figsize=(13,5))
df.shares.hist(bins=50)

#creating subplots for outliers
num_cols = df.select_dtypes(['int64','float64']).columns
num_plots = len(num_cols)
num_rows = math.ceil(num_plots / 13)  # You can adjust the number of columns as per your preference
num_cols_subplot = 5


fig, axes = plt.subplots(num_rows, num_cols_subplot, figsize=(15, 5 * num_rows))
axes = axes.flatten()

#Outliers
for i in range(len(num_cols)):
    sns.boxplot(df[num_cols[i]])
    plt.show()

for column in num_cols:    
    q1 = df[column].quantile(0.25)    # First Quartile
    q3 = df[column].quantile(0.75)    # Third Quartile
    IQR = q3 - q1                            # Inter Quartile Range

    llimit = q1 - 1.5*IQR                       # Lower Limit
    ulimit = q3 + 1.5*IQR                        # Upper Limit

    outliers = df[(df[column] < llimit) | (df[column] > ulimit)]
    print('Number of outliers in "' + column + '" : ' + str(len(outliers)))
    print(llimit)
    print(ulimit)
    print(IQR)

#scaling the dataset df2_num contains the numerical features while df2_cat contains the categorical features.

df2_num=df.drop(["monday","tuesday","wednesday","thursday",
                  "friday","saturday","sunday","is_weekend",                  
                  "lifestyle","entertainment","bus",
                  "socmed","tech","world"],axis=1)

df2_cat=df[["monday","tuesday","wednesday","thursday",
             "friday","saturday","sunday","is_weekend",            
             "lifestyle","entertainment","bus",
                  "socmed","tech","world"]]

#drop the target variable from df2_num

df2_num = df2_num.drop('shares',axis=1)

df2_num.columns

# Finding negative values

#negcols=df2_num.columns[(df2_num<=0).any()]
#negcols

#converting negative values to positive values

#for i in negcols:
 #   m=df2_num[i].min()
  #  name=i +'_new'
   # df2_num[name]=((df2_num[i]+1)-m)

#df2_num.columns

#for i in negcols:
 #   df2_num.drop(i,axis=1,inplace=True)

#negcols=df2_num.columns[(df2_num<=0).any()]
#negcols

pt=preprocessing.PowerTransformer(method='yeo-johnson',standardize=False)
df2_num_add=pt.fit_transform(df2_num)
df2_num_add=(pd.DataFrame(df2_num_add,columns=df2_num.columns))

# Treating outliers by capping values to a predefined range

for col in df2_num_add.columns:
    percentiles = df2_num_add[col].quantile([0.01,0.99]).values
    df2_num_add[col][df2_num_add[col] <= percentiles[0]] = percentiles[0]
    df2_num_add[col][df2_num_add[col] >= percentiles[1]] = percentiles[1]

num_cols = df2_num_add.select_dtypes(['int64','float64']).columns

for column in num_cols:    
    q1 = df2_num_add[column].quantile(0.25)    # First Quartile
    q3 = df2_num_add[column].quantile(0.75)    # Third Quartile
    IQR = q3 - q1                            # Inter Quartile Range

    llimit = q1 - 1.5*IQR                       # Lower Limit
    ulimit = q3 + 1.5*IQR                        # Upper Limit

    outliers = df2_num_add[(df2_num_add[column] < llimit) | (df2_num_add[column] > ulimit)]
    print('Number of outliers in "' + column + '" : ' + str(len(outliers)))
    print(llimit)
    print(ulimit)
    print(IQR)



##creating subplots for transformation
num_cols = df2_num_add.select_dtypes(['int64','float64']).columns
num_plots = len(num_cols)
num_rows = math.ceil(num_plots / 13)  # You can adjust the number of columns as per your preference
num_cols_subplot = 5


fig, axes = plt.subplots(num_rows, num_cols_subplot, figsize=(15, 5 * num_rows))
axes = axes.flatten()

#Boxplot transformation

for i in range(len(num_cols)):
    sns.boxplot(df2_num_add[num_cols[i]])
    plt.show()

df2_num_add.columns

df2_cat.columns

# Create a 'top_data_channel' column based on the data_channel columns
df2_cat['top_data_channel'] = df2_cat[['lifestyle', 'entertainment',
                             'bus', 'socmed',
                             'tech', 'world']].idxmax(axis=1)

# Print the unique values in the 'top_data_channel' column
unique_top_data_channels = df2_cat['top_data_channel'].unique()
print(f"Unique values in 'top_data_channel': {unique_top_data_channels}\n")

# Define a function to extract the last word from a string
def extract_last_word(channel):
    words = channel.split('_')
    return words[-1]

# Apply the function to the 'top_data_channel' column and create a new 'top_data_channel_last_word' column
df2_cat['top_data_channel_last_word'] = df2_cat['top_data_channel'].apply(extract_last_word)

# Print the unique values in the 'top_data_channel_last_word' column
unique_last_words = df2_cat['top_data_channel_last_word'].unique()
print(f"Unique last words in 'top_data_channel_last_word': {unique_last_words}\n")
df2_cat = df2_cat.drop('top_data_channel',axis=1)

df_final=pd.concat([df2_num_add,df2_cat],axis=1)

df_final.shape

df_final['popularity'] = df['shares'].apply(lambda x: 0 if x <1400 else 1)

df_final.isnull().sum()

df_final=df_final.dropna()

df_final.columns

df_final.shape

