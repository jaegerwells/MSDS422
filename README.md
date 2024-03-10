# MSDS422 Group Project

this is our Github for MSDS422 project for Winter 2024 quarter

For Python code, please see **group_project_v3.ipynb** for all code and visuals:

https://github.com/jaegerwells/MSDS422/blob/76123d38dff0c923e96c79a8ae83502a91eccd5e/group_project_v3.ipynb 

##### MSDS 422 Winter 2024

##### Conor Walsh, Albert Lee, Yueh-Chang Kuo, Jaeger Wells

## Executive Summary

Online news and social media have become a dominant force in our society today. "Going viral" or shareability is how people learn about world events. While news feels like it is unbalanced and incredible negative most of the time, there is a need to explore what are the attributes of articles that drive shareability. By utilizing data from over 2 years from the website Mashable, we will look to understand what are the characteristics of a news article going viral. 

## Problem Statement / Research Objectives

Our problem statement is simple; What are the characteristics that make news go viral? We will look to understand if subjectivity, overall polarity, channel news is being displayed, and type of file format for news impacts the total number of shares on Mashable. Ultimately, we are view this analysis as a binary classification problem.

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

When thinking about news sentiment, it appears that there is a fairly normal distribution in the number of shares of news articles. While not skewed overly negative, there is a distribution that has the majority of data points falling within -0.2 and 0.4 sentiment, which has a longer tail on the positive sentiment spectrum. While the sentiment shows slightly more positive, looking at the distribuition of articles by subjectivity shows that the distribution of articles are disributed in a range that is fairly subjective. Taking these datapoints into consideration, it could be thought of that highly subjective, fairly positive articles are more likely to be viral.

 We also wanted to initially look at the count of popular vs. unpopular over different channels on Mashable. Lifestyle, Business, Social Media, and Tech are more popular vs. Entertainment and World are more unpopular.

 
## Feature Engineering

One of the key things that we noticed through also through the EDA process was that there were not many strong correlations to the target variable "shares"; which is the target for number of times an article was shared. This coupled with the number of significant outliers in the distribution of the data set. We performed a Yeo-Johnson transformation to the variables in order to normalize distribution in order to handle the negative values within the dataset (Hvitfeldt 2024), as well as mitigating the impact of the outliers within the dataset.

# Methodology and tools used in the process
## Tooling

We have primarily been able to utilize a Python (Jupyter) to conduct our code and utilizing Python libraries such as Pandas, Numpy, Scikitlearn, and Seaborn in the aid of conducting this analysis. We have used Github as a repository for the project to aid in providing a total solution as well as version control as we work in a distributed space. By enabling the analysis to run in a virtual environment helps for reproducibility if other people would like to replicate and run our analysis on their own systems.

## Hardware

The primary hardware that we have utilized have been our respective CPU processors for local analysis by each member of the team. Because we have not used any deep learning techniques in this analysis such as neural networks we have not had to utilize GPU or TPUs to aid in the analysis.

## Model Selection

In light of understanding the problem statement of determining the characteristics of viral news, we aim to understand this in a classification analysis as opposed to a regression analysis. Since ultimately we are interested in understanding the attributes of viral news, we selected the following model types to conduct our analysis:

```
* Naive Bayes Gaussian Classifier
* Logistic Classification
* Random Forest Classifier
* KNeighborsClassifier
```

We evaluate each of these models in afew different ways. The first look at a confusion matrix to look at accuracy, such as the instances of True Positives/True Negatives and False Positives/False Negatives. We also are able to compare each type of model by producing ROC/AUC curves.

## Model deployment strategy

While we have not fully deployed the models in question, but we would follow industrty standard approaches to machine learning model deployment. This includes first choosing the deployment environment best suited to the needs of the project and other requirements. This could be utilizing environments such as AWS, Azure, or Google Cloud.

Next we would have to package the model and it's dependencies into a container and then deploy the container. Containerization methods allow for a model, runtime environment, and it's dependencies are packaged together to ensure reproducibility. From there we would need to set up monitoring to ensure there aren't any issues and maintanence (Logunva 2023).

# Findings and Conclusions

Using Binary Classification to binarize the data into popular vs. unpopular news categories. The median threshold was 1,400 shares. From there we performed 4 models as discussed in slide 7. Using Accuracy as the performance metric, we were able to determine that the Random Forest model performed the best across all sample datasets. We then hyper tuned the model to increase the overall accuracy up to ~67%.

While we were able to predict with ~67% accuracy there is room to continue to improve the models. The random forest model classification before hyper tuning was 66.39% and after hyper tuning increased to 66.99%. With a 67% accuracy, the business should consider using these models to drive increased shares. Surprisingly, our accuracy was also in line with the original authors in their paper - “A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News”.

The 67% accuracy highlights variability with predictions also depends on the dataset. The imbalance in the class distribution caused the model to be more biased towards popular articles.

As we reflect, on our analysis, we could have included other variables such as word count or certain keywords and factor that into the performance of the models.
 
Another interesting data that we would have liked to seen from the dataset is around variables on the author, author experience, number of articles that authors have published, etc. One possibility is that certain authors may perform better than others.

# Lessons Learned and Recommendations
## Lessons Learned

We have learned several key things in this analysis. First, we have found that a hypertuned RandomForest model is a feasible solution for predicting news virality. We have found that this is validated from other studies including Johnson & Weinberger (n.d.) and Fernandes et al (2015). All of the classification machine learning models utilized were able to achieve greater than 55% accuracy, and we believe that this could be improved upon with additional data sources from outside of Mashable.

Second, we can limit the impact of outliers through feature engineering. By limiting outliers helps de-risk overfitting a model to training data and allows for higher performance of unseen data.


## Recommendations

There are several recommendations as we think about deploying this analysis. First, explore training models on smaller percentages of datasets. Our analysis indicates <3% change in performance across all models when training with 10% and 100% of the dataset. This could also save on computation time; training excursions may be run on smaller percentages of the dataset to determine a baseline model feasibility. 

Another recommendation for further analysis would be to utilize this analysis as a jumping off point to incorporate a multi label classification analysis, so better understand what goes viral for each of the channel types (Jain 2017). This could help Mashable fine tune their own sharing algorithms to continue to supercharge engagement.

### Potential Third Party Datasets

While Mashable provides a robust dataset to utilize for this analysis, it is just one source of news that we are looking at. In order to expand this analysis into something of a larger scale, we would need to leverage third party datasets to get a more holistic view of digital news consumption and virality. One of these datasets can be found from Meneame, which is a Spanish digital news source. This data set is available on Kaggle:

https://www.kaggle.com/datasets/thedevastator/popular-news-articles-popularity-on-meneame

# References

Logunva, I. (2023). ML model deployment: Challenges, solutions & best practices. Retrieved from https://serokell.io/blog/ml-model-deployment

Holtz, N. (2023). What is CRISP DM?. Data Science Process Alliance. Retrieved from https://www.datascience-pm.com/crisp-dm-2/ 

Hvitfeldt, E. (2024). Numeric Transformation using Yeo-Johnson Transformation. In Feature Engineering A-Z. Retrieved from https://feaz-book.com/numeric-yeojohnson

Johnson, A., & Weinberger, D. (n.d.). Predicting News Sharing on Social Media. Stanford CS229 Project Reports. Retrieved from https://cs229.stanford.edu/proj2016/report/JohnsonWeinberger-PredictingNewsSharing-report.pdf

K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News. Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence, September, Coimbra, Portugal. Retrieved from https://archive.ics.uci.edu/dataset/332/online+news+popularity 

Jain, S. (207). Introduction to Multi-label Classification. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/
