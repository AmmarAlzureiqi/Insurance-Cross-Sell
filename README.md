# Insurance-Cross-Sell-ML

# Customersʼ Interest on Vehicle Insurances

## Introduction


An Insurance company that has provided Health Insurance to its customers wishes to build a model to predict whether the policyholders (customers) from the past year will also be interested in Vehicle Insurance provided by the company.

Just like medical insurance, there is vehicle insurance where every year a customer needs to pay a premium of a certain amount to the insurance company so that in case of an unfortunate accident by the person driving the vehicle, the insurance company will provide compensation (called ʻsum assuredʼ) to the customer.

Building a model to predict whether a customer would be interested in Vehicle Insurance is extremely helpful for the company because it can then accordingly plan its communication strategy to reach out to those customers and optimize its business model and revenue.

## Dataset description

The dataset is available on Kaggle, provided by one whose client is an Insurance company. There are three datasets available: train.csv, test.csv, and sub.csv. The test.csv file is for prediction and does not include a response variable while the sub.csv file is for submission of the predicted response. Since we are only interested in the relationship between the response and the predictors, and not the predictions themselves, we have only made use of the train.csv file, which contains the necessary data.

The dataset called train.csv includes the following 12 variables:

<p align="center">
<img src='https://github.com/AmmarAlzureiqi/Insurance-Cross-Sell-ML/assets/100096699/84bac6e7-8f52-4c6f-9168-ad1d411b72f9' width='100%' height='300'>
</p>
<p align="center">
<img src='https://github.com/AmmarAlzureiqi/Insurance-Cross-Sell-ML/assets/100096699/104ea7b2-1e4d-4eea-9813-d43f5d4ba2ad' width='100%' height='300'>
</p>

The data includes 266,776 observations with no missing values. 88% of the customers are not interested in the vehicle while others are interested. 46% of the customers are female while others are male. The customers are aged from 20 to 85, and a median of 36.54% of the customers have previously insured while the others have not. Almost every customer has a driving license.

## Methods

Classification methods require supervised machine learning algorithms to assign a class/label to a set of data points. Some common algorithms include logistic regression, LDA, QDA, and classification trees. In this project, the goal is to classify a policyholder into one of two classes while the models will use various factors to predict if an individual may be interested in vehicle insurance.

<ins>Confusion Matrix and Classification Statistics:</ins>

In binary classification problems, confusion matrices are used as important indicators of model performance. These matrices compare the actual classifications with the predicted classifications and display the counts of correctly and incorrectly classified observations. The values on the diagonal represent the counts of correctly classified values, while the remaining values represent the counts of incorrectly classified observations.
Additionally, some common statistics that are calculated from these values are used to assess model performance, which include:

⦁ Accuracy: The proportion of observations that were classified correctly
⦁ Specificity: The proportion of negatives that are correctly classified
⦁ Sensitivity (Recall): The proportion of positives that are correctly classified
⦁ Precision: The proportion of predicted positives that were correctly classified
⦁ F1 Score: The harmonic mean between recall and precision
 
### Logistic Regression:

Logistic regression models, similarly to linear regression models, use a linear combination of the predictors to predict the response. However, instead of estimating the response directly, logistic regression uses the logit link function that links the linear combination of predictors to the estimated probability of having a ʻpositiveʼ response.

### Decision Trees:
A non-parametric supervised machine learning algorithm, which can be used in both regression and classification problems. The general structure of a decision tree is as follows; the root node is the top of the decision tree, as we descend down the tree, each node where a decision must be made (and the predictor space is split) is called an internal node. At the ends of the tree we have the terminal nodes that correspond to a subset of the predictor space, and these provide the predicted value, or label the ones that are to be assigned to predictors that are in those respective subsets. Decision trees result in highly interpretable models that are very simple to explain.

### Boosting and Bagging:

Boosting of decision trees is used to improve predictive capabilities by growing multiple trees sequentially. Instead of training one large decision tree (strong classifier) and risking overfitting, the algorithm begins with much smaller trees (weak classifiers) that are fitted and incorporated in training subsequent trees so that each tree essentially learns from the information provided by the tree(s) that were fitted before it. In essence, the boosted model will learn ʻslowlyʼ as more trees are fit. Gradient boosting is a type of boosting technique where each tree is specifically evaluated based on its loss function (which can be chosen by the user), by using gradient descent optimization.

### LDA and QDA:

Linear discriminant analysis is similar to logistic regression, but instead of modelling P(Y=k|X=x) using the logit link, LDA attempts to model the distribution of the predictors for each class, using Bayes theorem. QDA is similar to LDA except the covariance matrix depends on the class k.
We fitted models using logistic regression, linear discriminant analysis, and quadratic discriminant analysis. We found the best thresholds with ROC analysis for the test data. We applied Synthetic Minority Over-sampling Technique (SMOTE) and simple undersampling to obtain balanced datasets. We transferred the factors to dummy variables and built some k-Nearest-Neighbour classifiers from the data. We also applied simple decision trees, random forests, and boostings to the data. The final decision we made was based on the sensitivity, specificity, and accuracy of each model.

### A Brief Introduction to SMOTE

When a dataset is unbalanced, like in our case, that nearly 90% of the responses are 0 (customers not interested in the new vehicle insurance). The predictions of models are likely to be affected by the majority class too much. For instance, our first tree model turned out to produce only 0 predictions - that means the model did not classify the data at all. Resampling is a useful method to fix this problem. We can oversample class 1 or undersample class 0 to obtain a balanced dataset and use this new dataset to train our model.
SMOTE is a method that combines over-sampling the minority (abnormal) class and under-sampling the majority (normal) class, introduced by Nitesh V. Chawla, Kevin W. Bowyer, and their team in 2002. 

SMOTE can achieve better classifier performance (in ROC space) than only under-sampling the majority class.

### Results and Analysis are displayed in PDF file

## Conclusion

This dataset is strongly unbalanced and one can easily obtain a wrong accuracy of 0.88, which is the proportion of the response 0. We used some resampling methods to obtain a balanced dataset to train our models. And the best model we found was kNN with undersampling data, whose sensitivity, specificity, and accuracy are all high, but this method is relatively slow in the running speed.

