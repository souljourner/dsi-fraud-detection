# Ecommerce Fraud Detection
Group project by Jeff Li, Lihua Ma, Jeff Sang and John Zhu

## Scope
There were a myriad of ways we could've approached our project. The first question we asked ourselves was:

How the heck do we define fraud?

In our dataset, under the acct_type, there were three different versions of fraud and another three different versions of "spam." We assumed "premium" accounts were "Not Fraud." We decided to remove any account that contained "spam" from our dataset, since we didn't have access to company management/know the company's business goals. We did this with the awareness that we may be removing a good chunk of data points from the dataset, which could affect our results. 

## Data Cleaning
To clean the data, there were a few things we had to do:

1. Transform categorical variables into their numerical equivalent.

2. Set aside text-heavy variables.

3. Convert time-stamp variables into a read-able date format. 

Yes, for columns like "description", there is a possibility of seeing some signal in there. However, our team felt it best to get a model up and running before doing any additional NLP analysis on text columns. 

## Feature-Engineering
We tried a variety of approaches to Feature-Engineering. The entire dataset had about 44 features built-in. We tried two different methods on the existing features:

1. Removing features that require further cleaning(text-heavy columns), and then running the model through the rest of the features. 

2. Pre-selecting features that we believed, would feed a model quality data. 

In addition, we decided engineer a number of new features:

1. Previous Payout Total: How many times did the customer pay out? We felt that previous payout total had some signal predicting fraud.

In our final model, a six feature-matrix seemed to perform the best:

'delivery_method'
'num_payouts'
'org_facebook'
'org_twitter'
'sale_duration'
'previous_payouts_total'

## Modeling

For modeling, we tried a variety of different models:

- Logistic Regression

- Gaussian Process

- Nearest Neighbors

- Linear SVM

- Decision Tree

- Random Forest

- AdaBoost

- Gradient Boost

- Neural Networks

- Naives Bayes

- QDA

And out of all the models, the Random Forest gave us the best AUC score at 98%. 

## Scoring

There were a variety of methods we considered for scoring, from a technical standpoint, we used:

- AUC

- Accuracy

- Precision

- Recall

From a business standpoint, we also wanted to consider the different valuations for different customers. As a result, we used a Customer Lifetime Value calculation to calculate lifetime value of each customer. We also assumed that our company, took 5% of the ticket sales as commission for sale. 

## Results

As a result, we ended with a 99% AUC score and 98% accuracy using a Gradient Boosting Classifier. 
