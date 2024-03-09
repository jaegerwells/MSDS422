# MSDS422 Group Project

this is our Github for MSDS422 project for Winter 2024 quarter

For Python code, please see **group_project_v3.ipynb** for all code and visuals

#### MSDS 422 Winter 2024

#### Conor Walsh, Albert Lee, Yueh-Chang Kuo, Jaeger Wells

## Executive Summary

Online news and social media have become a dominant force in our society today. "Going viral" or shareability is how people learn about world events. While news feels like it is unbalanced and incredible negative most of the time, there is a need to explore what are the attributes of articles that drive shareability. By utilizing data from over 2 years from the website Mashable, we will look to understand what are the characteristics of a news article going viral. 

## Problem Statement / Research Objectives

Our problem statement is simple; What are the characteristics that make news go viral? We will look to understand if subjectivity, overall polarity, channel news is being displayed, and type of file format for news impacts the total number of shares on Mashable. Ultimately, we are viewing this analysis as a binary classification problem.

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

One of the key things that we noticed through also through the EDA process was that there were not many strong correlations to the target variable "shares"; which is the target for number of times an article was shared. This coupled with the number of significant outliers in the distribution of the data set. We performed a Yeo-Johnson transformation to the variables in order to normalize distribution in order to handle the negative values within the dataset (Hvitfeldt 2024), as well as mitigating the impact of the outliers within the dataset.

# Methodology and tools used in the process

## Tooling

We have primarily been able to utilize a Python (Jupyter) to conduct our code and utilizing Python libraries such as Pandas, Numpy, Scikitlearn, and Seaborn in the aid of conducting this analysis. We have used Github as a repository for the project to aid in providing a total solution as well as version control as we work in a distributed space. By enabling the analysis to run in a virtual environment helps for reproducibility if other people would like to replicate and run our analysis on their own systems.

## Hardware

the primary hardware that we have utilized have been our respective CPU processors for local analysis by each member of the team. Because we have not used any deep learning techniques in this analysis such as neural networks we have not had to utilize GPU or TPUs to aid in the analysis.

## Model Selection

In light of understanding the problem statement of determining the characteristics of viral news, we aim to understand this in a classification analysis as opposed to a regression analysis. Since ultimately we are interested in understanding the attributes of viral news, we selected the following model types to conduct our analysis:

```
*Naive Bayes Gaussian Classifier
*Logistic Classification
*Random Forest
*KMeans Nearest Neighbor
```

We evaluate each of these models in afew different ways. The first look at a confusion matrix of each type of model as well as producing ROC/AUC curves to understand accuracy.

## Model deployment strategy

While we have not fully deployed the models in question, but we would follow industrty standard approaches to machine learning model deployment. This includes first choosing the deployment environment best suited to the needs of the project and other requirements. This could be utilizing environments such as AWS, Azure, or Google Cloud.

Next we would have to package the model and it's dependencies into a container and then deploy the container. Containerization methods allow for a model, runtime environment, and it's dependencies are packaged together to ensure reproducibility. From there we would need to set up monitoring to ensure there aren't any issues and maintanence (Logunva 2023).

# Findings and Conclusions

TBD

# Lessons Learned and Recommendations

## Lessons Learned

## Recommendations

One key recommendation for further analysis would be to utilize this analysis as a jumping off point to incorporate a multi label classification analysis, so better understand what goes viral for each of the channel types. This could help Mashable fine tune their own sharing algorithms to continue to supercharge engagement.

### Potential Third Party Datasets

While Mashable provides a robust dataset to utilize for this analysis, it is just one source of news that we are looking at. In order to expand this analysis into something of a larger scale, we would need to leverage third party datasets to get a more holistic view of digital news consumption and virality. One of these datasets can be found from Meneame, which is a Spanish digital news source. This data set is available on Kaggle:

https://www.kaggle.com/datasets/thedevastator/popular-news-articles-popularity-on-meneame

# References

Logunva, I. (2023). ML model deployment: Challenges, solutions & best practices. Retrieved from https://serokell.io/blog/ml-model-deployment

Hvitfeldt, E. (2024). Numeric Transformation using Yeo-Johnson Transformation. In Feature Engineering A-Z. Retrieved from https://feaz-book.com/numeric-yeojohnson

Johnson, A., & Weinberger, D. (n.d.). Predicting News Sharing on Social Media. Stanford CS229 Project Reports. Retrieved from https://cs229.stanford.edu/proj2016/report/JohnsonWeinberger-PredictingNewsSharing-report.pdf
