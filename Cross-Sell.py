import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
import seaborn as sns
import statsmodels.api as sm
from plotnine import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import tree
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report 
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
from xgboost import XGBClassifier

df = pd.read_csv('train.csv')
# df.drop(['id', 'Region_Code', 'Policy_Sales_Channel'], axis = 1, inplace = True)
df.drop(['id'], axis = 1, inplace = True)
df1 = df

df.isnull().values.any() # no null values

count_no = sum(df['Response']==0)
n = sum(df['Response'].value_counts())
print("number of observations:", n)
print("percentage of not interested", count_no/n*100)
print("percentage of interested", (1 - count_no/n)*100)

print("customer age ranges from ", df['Age'].min(), 'to', df['Age'].max())
print("propertion of male clients:", sum(df['Gender'] == 'Male')/n)
print("proportion of clients that already have vehicle insurance:", sum(df['Previously_Insured'])/n)
print("proportion of clients with a drivers license:", sum(df['Driving_License'])/n)


df['Response'].value_counts() # unbalanaced data
# Creating Bar Plot
sns.countplot(x = 'Response', data = df)
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.5)
fig.add_subplot(2, 2, 1)
sns.countplot(x = 'Gender', data = df)
fig.add_subplot(2, 2, 2)
sns.countplot(x = 'Driving_License', data = df)
fig.add_subplot(2, 2, 3)
sns.countplot(x = 'Previously_Insured', data = df)
fig.add_subplot(2, 2, 4)
sns.countplot(x = 'Vehicle_Damage', data = df)
plt.show()

sns.countplot(x = 'Vehicle_Age', data = df)
plt.show()

(pd.crosstab(df['Response'],df['Vehicle_Damage'], 
             normalize='index')
   .plot.bar(stacked=True)
)
(pd.crosstab(df['Vehicle_Age'], df['Response'], 
             normalize='index')
   .plot.bar(stacked=True)
)

(pd.crosstab(df['Response'],df['Previously_Insured'], 
             normalize='index')
   .plot.bar(stacked=True)
)
(pd.crosstab(df['Response'],df['Gender'], 
             normalize='index')
   .plot.bar(stacked=True)
)

sns.kdeplot(df.loc[df['Response'] == 1, 'Age'], linewidth=2, fill=True, label = "Interested in Insurance")
sns.kdeplot(df.loc[df['Response'] == 0, 'Age'], linewidth=2, fill=True, label = "Not Interested in Insurance")
plt.legend(loc="upper right")
plt.xlim(15,85)

sns.kdeplot(df.loc[df['Response'] == 1, 'Annual_Premium'], linewidth=2, fill=True, label = "Interested in Insurance")
sns.kdeplot(df.loc[df['Response'] == 0, 'Annual_Premium'], linewidth=2, fill=True, label = "Not Interested in Insurance")
plt.legend(loc="upper right")
plt.xlim(0,125000)

sns.kdeplot(df.loc[df['Response'] == 1, 'Vintage'], linewidth=2, fill=True, label = "Interested in Insurance")
sns.kdeplot(df.loc[df['Response'] == 0, 'Vintage'], linewidth=2, fill=True, label = "Not Interested in Insurance")
plt.legend(loc="upper right")

le = LabelEncoder() 
df['Gender'] = le.fit_transform(df['Gender']) 
df['Driving_License'] = le.fit_transform(df['Driving_License']) 
df['Previously_Insured'] = le.fit_transform(df['Previously_Insured']) 
df['Vehicle_Age'] = le.fit_transform(df['Vehicle_Age']) 
df['Vehicle_Damage'] = le.fit_transform(df['Vehicle_Damage']) 
df['Policy_Sales_Channel'] = le.fit_transform(df['Policy_Sales_Channel']) 
df['Region_Code'] = le.fit_transform(df['Region_Code'])
df['Response'] = le.fit_transform(df['Response'])

df['Gender'] = df['Gender'].astype('category')
df['Driving_License'] = df['Driving_License'].astype('category')
df['Previously_Insured'] = df['Previously_Insured'].astype('category')
df['Vehicle_Age'] = df['Vehicle_Age'].astype('category')
df['Vehicle_Damage'] = df['Vehicle_Damage'].astype('category')
df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('category')
df['Region_Code'] = df['Region_Code'].astype('category')
df['Response'] = df['Response'].astype('category')

y = df['Response']
X = df.drop(['Response'], axis = 1)

df.dtypes

le = LabelEncoder() 
df1['Gender'] = le.fit_transform(df1['Gender']) 
df1['Driving_License'] = le.fit_transform(df1['Driving_License']) 
df1['Previously_Insured'] = le.fit_transform(df1['Previously_Insured']) 
df1['Vehicle_Age'] = le.fit_transform(df1['Vehicle_Age']) 
df1['Vehicle_Damage'] = le.fit_transform(df1['Vehicle_Damage']) 
df1['Region_Code'] = le.fit_transform(df1['Region_Code'])
df1['Policy_Sales_Channel'] = le.fit_transform(df1['Policy_Sales_Channel'])
df1['Response'] = le.fit_transform(df1['Response'])
y2 = df1['Response']
X2 = df1.drop(['Response'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=32)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.33, random_state=32)

logistic_model = LogisticRegression(max_iter = 1000).fit(X_train,y_train)

y_pred = logistic_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
Sensitivity = Recall = cm[1,1] / (cm[1,1] + cm[1,0])
Specificity = cm[0,0] / (cm[0,0] + cm[0,1])

print ("Accuracy : ", round(accuracy_score(y_pred, y_test),3))
print("Sensitivity : ", round(Sensitivity,5)) 
print("Specificity : ", round(Specificity,5))
print("Precision : ", round(precision_score(y_test, y_pred), 3))
print("F1 Score: ", round((2*precision_score(y_test, y_pred)*Recall)/(precision_score(y_test, y_pred) + Recall), 3))
print("AUC : ", round(roc_auc_score(y_test, logistic_model.predict(X_test)), 3))

print(classification_report(y_test, y_pred))
plt.figure(figsize=(10,5))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5,cmap='tab20')
plt.show()

X_arr = np.array(X)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_arr, y, test_size = 0.2, random_state = 2)

os = SMOTE(random_state=0)

os_data_X, os_data_y = os.fit_resample(X_train1, y_train1)
os_data_X = pd.DataFrame(data = os_data_X, columns = X_train.columns)

os_data_X2, os_data_y2 = os.fit_resample(X_train2, y_train2)
os_data_X2 = pd.DataFrame(data = os_data_X2, columns = X_train2.columns)

logistic_model_os = LogisticRegression(max_iter=1000).fit(os_data_X, os_data_y)
y_pred_os = logistic_model_os.predict(X_test)

y_pred = logistic_model_os.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
Sensitivity = Recall = cm[1,1] / (cm[1,1] + cm[1,0])
Specificity = cm[0,0] / (cm[0,0] + cm[0,1])

print ("Accuracy : ", round(accuracy_score(y_pred, y_test),3))
print("Sensitivity : ", round(Sensitivity,5)) 
print("Specificity : ", round(Specificity,5))
print("Precision : ", round(precision_score(y_test, y_pred), 3))
print("F1 Score: ", round((2*precision_score(y_test, y_pred)*Recall)/(precision_score(y_test, y_pred) + Recall), 3))
print("AUC : ", round(roc_auc_score(y_test, logistic_model_os.predict(X_test)), 3))
print(classification_report(y_test, y_pred))
plt.figure(figsize=(10,5))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5,cmap='tab20')
plt.show()

data_final_vars = df.columns.values.tolist()
logreg = LogisticRegression()
rfe = RFE(logistic_model_os, n_features_to_select = 4, step = 2)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(X.columns[rfe.support_])


os_data_rfe = os_data_X[os_data_X.columns[rfe.support_]]
X_test_rfe = X_test[X_test.columns[rfe.support_]]

logistic_model_os_rfe = LogisticRegression().fit(os_data_rfe, os_data_y)

y_pred = logistic_model_os_rfe.predict(X_test_rfe)

cm = confusion_matrix(y_test, y_pred)
Sensitivity = Recall = cm[1,1] / (cm[1,1] + cm[1,0])
Specificity = cm[0,0] / (cm[0,0] + cm[0,1])

print ("Accuracy : ", round(accuracy_score(y_pred, y_test),3))
print("Sensitivity : ", round(Sensitivity,5)) 
print("Specificity : ", round(Specificity,5))
print("Precision : ", round(precision_score(y_test, y_pred), 3))
print("F1 Score: ", round((2*precision_score(y_test, y_pred)*Recall)/(precision_score(y_test, y_pred) + Recall), 3))

print("AUC : ", round(roc_auc_score(y_test, logistic_model_os_rfe.predict(X_test_rfe)), 3))
print(classification_report(y_test, y_pred))
plt.figure(figsize=(10,5))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5,cmap='tab20')
plt.show()

clf_tree = DecisionTreeClassifier().fit(os_data_X, os_data_y)

y_pred = clf_tree.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
Sensitivity = Recall = cm[1,1] / (cm[1,1] + cm[1,0])
Specificity = cm[0,0] / (cm[0,0] + cm[0,1])

print ("Accuracy : ", round(accuracy_score(y_pred, y_test),3))
print("Sensitivity : ", round(Sensitivity,5)) 
print("Specificity : ", round(Specificity,5))
print("Precision : ", round(precision_score(y_test, y_pred), 3))
print("F1 Score: ", round((2*precision_score(y_test, y_pred)*Recall)/(precision_score(y_test, y_pred) + Recall), 3))
print("AUC : ", round(roc_auc_score(y_test, y_pred), 3))
print(classification_report(y_test, y_pred))
plt.figure(figsize=(10,5))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5,cmap='tab20')
plt.show()

rfc_os = RandomForestClassifier().fit(os_data_X , os_data_y)

y_pred = rfc_os.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
Sensitivity = Recall = cm[1,1] / (cm[1,1] + cm[1,0])
Specificity = cm[0,0] / (cm[0,0] + cm[0,1])

print ("Accuracy : ", round(accuracy_score(y_pred, y_test),3))
print("Sensitivity : ", round(Sensitivity,5)) 
print("Specificity : ", round(Specificity,5))
print("Precision : ", round(precision_score(y_test, y_pred), 3))
print("F1 Score: ", round((2*precision_score(y_test, y_pred)*Recall)/(precision_score(y_test, y_pred) + Recall), 3))
print ("AUC : ", round(roc_auc_score(y_test, y_pred),5))
print(classification_report(y_test, y_pred))
plt.figure(figsize=(10,5))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5,cmap='tab20')
plt.show()

xg = xgb.XGBClassifier(n_estimators = 1000)
evaluation = [( X_train2, y_train2), ( X_test2, y_test2)]
xg.fit(X_train2, y_train2, eval_set=evaluation, eval_metric="auc", 
       early_stopping_rounds=20, verbose=100)

y_pred = xg.predict(X_test2)
cm = confusion_matrix(y_test2, y_pred)
Sensitivity = Recall = cm[1,1] / (cm[1,1] + cm[1,0])
Specificity = cm[0,0] / (cm[0,0] + cm[0,1])

print ("Accuracy : ", round(accuracy_score(y_pred, y_test2),3))
print("Sensitivity : ", round(Sensitivity,5)) 
print("Specificity : ", round(Specificity,5))
print("Precision : ", round(precision_score(y_test, y_pred), 3))
print("F1 Score: ", round((2*precision_score(y_test, y_pred)*Recall)/(precision_score(y_test, y_pred) + Recall), 3))
print ("AUC : ", round(roc_auc_score(y_test, y_pred),5))
print(classification_report(y_test, y_pred))
plt.figure(figsize=(10,5))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5,cmap='tab20')
plt.show()

evaluation = [( os_data_X2, os_data_y2), ( X_test2, y_test2)]
xg.fit(os_data_X2, os_data_y2, eval_set=evaluation, eval_metric="auc", 
       early_stopping_rounds=20, verbose=100)

y_pred = xg.predict(X_test2)

cm = confusion_matrix(y_test2, y_pred)
Sensitivity = Recall = cm[1,1] / (cm[1,1] + cm[1,0])
Specificity = cm[0,0] / (cm[0,0] + cm[0,1])

print ("Accuracy : ", round(accuracy_score(y_pred, y_test),3))
print("Sensitivity : ", round(Sensitivity,5)) 
print("Specificity : ", round(Specificity,5))
print("Precision : ", round(precision_score(y_test, y_pred), 3))
print("F1 Score: ", round((2*precision_score(y_test, y_pred)*Recall)/(precision_score(y_test, y_pred) + Recall), 3))
print("AUC : ", round(roc_auc_score(y_test, y_pred), 3))
print(classification_report(y_test, y_pred))
plt.figure(figsize=(10,5))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5,cmap='tab20')
plt.show()


