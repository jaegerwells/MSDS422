# MSDS422 Group Project
this is our Github for MSDS422 project for Winter 2024 quarter

#### MSDS 422 Winter 2024
#### Conor Walsh, Albert Lee, Yueh-Chang Kuo, Jaeger Wells


## Executive Summary

Online news and social media have become a dominant force in our society today. "Going viral" or shareability is how people learn about world events. While news feels like it is unbalanced and incredible negative most of the time, there is a need to explore what are the attributes of articles that drive shareability. By utilizing data from over 2 years from the website Mashable, we will look to understand what are the characteristics of a news article going viral.

## Problem Statement / Research Objectives

Our problem statement is simple; What are the characteristics that make news go viral? We will look to understand if subjectivity, overall polarity, channel news is being displayed, and type of file format for news impacts the total number of shares on Mashable.


## Dataset Citation

K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision
    Support System for Predicting the Popularity of Online News. Proceedings
    of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence,
    September, Coimbra, Portugal.

## Dataset

https://archive.ics.uci.edu/dataset/332/online+news+popularity

# EDA and Feature Engineering

## Exploratory Data Analysis (EDA)

In this dataset, there are 62 total columns with roughly 39,650 rows of data. the majority of the variables within this dataset are numerical, with specific variables such as num_non_stop_words and avg_positive_polarity, being hallmarks of natural language processing models (NLP). 

Throughout our EDA, we’re able to determine the count of popular vs. unpopular news over different days of the week. News is more popular Monday, Thursday, Friday, Saturday, and Sunday. Whereas the news is more unpopular on Tuesday and Wednesday. 

We also wanted to initially look at the count of popular vs. unpopular over different channels on Mashable. Lifestyle, Business, Social Media, and Tech are more popular vs. Entertainment and World are more unpopular.

## Feature Engineering

One of the key things that we noticed through also through the EDA process was that there were not many strong correlations to the target variable "shares"; which is the target for number of times an article was shared. This coupled with the number of significant outliers in the distribution of the data set. We performed a Yeo-Johnson transformation to the variables in order to normalize distribution, as well as mitigating the impact of the outliers within the dataset.