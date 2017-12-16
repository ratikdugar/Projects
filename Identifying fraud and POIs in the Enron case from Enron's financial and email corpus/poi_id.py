
# coding: utf-8

# # Identifying Fraud From Enron Emails and Financial Data
# ##### - By Ratik Dugar

# ### In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, I  will build a person of interest identifier based on financial and email data made public as a result of the Enron scandal. We've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity. 
# ### During the course of this project, I will be using machine learning and other data analysis techniqes to clean and visualize the data. I will also explore various algorithms to build an identifier that performs as accurately as possible.

# ## Data overview and exploration

# #### Below I load the data and select the features that I want to use and explore. I then visualize the salaries and bonuses of everyone in the dataset to identify possible outliers and remove ones that are not valid points.

# In[197]:

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary',
                 'bonus',
                 'director_fees',
                 'exercised_stock_options',
                 'expenses',
                 'loan_advances',
                 'long_term_incentive',
                 'other',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'total_payments',
                 'total_stock_value',
                 'from_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi',
                 'to_messages'] 

### Load the dictionary containing the dataset
with open("../final_project/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print 'The number of observations:',len(data_dict)
print 'The number of features:',  len(features_list)

### Task 2: Remove outliers
data = featureFormat(data_dict, features_list)
#visualize in order to identify possible outliers
for point in data:
    salary = point[1]
    bonus = point[2]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.title("Data before removal")
plt.show()

data_dict.pop("TOTAL",0)
data = featureFormat(data_dict, features_list)

#visualization after removing the outlier above
for point in data:
    salary = point[1]
    bonus = point[2]
    plt.scatter( salary, bonus )
plt.xlabel("salary")
plt.ylabel("bonus")
plt.title("Data after removal")
plt.show()


# #### After removing the outlier, our scatterplot looks a little more contained but there still seems to be 4 points that are a little out of the way from the rest. We will explore those to make sure they are valid. 

# In[198]:

outliers = []
for key in data_dict:
    sal = data_dict[key]['salary']
    if sal == 'NaN':
        continue
    outliers.append((key, int(sal)))

outliers_info = (sorted(outliers,key=lambda x:x[1],reverse=True)[:4])
### print top 4 salaries
print outliers_info


# #### The 4 other distant points are from actual people who worked at Enron with exceptionally high salaries and so we decide to keep these valid points in our dataset.

# #### Below we find some more details about the Enron dataset.

# In[199]:

for person_name in data_dict:
    if data_dict[person_name]['salary']=='NaN':
        print person_name
print
poicount=0
for person_name in data_dict:
    if data_dict[person_name]['poi']==1.:
        poicount+=1
print "Number of POIs in the dataset:",poicount

#drop "THE TRAVEL AGENCY IN THE PARK" as that is not a person
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)
print "Number of entries in the dataset after removals:",len(data_dict)
print "Number of innocent employees in the dataset:",(len(data_dict)-poicount)


# #### From our information above, I find that there are 18 POIs in this dataset. After removing the outlier and the invalid data entry "THE TRAVEL AGENCY IN THE PARK', we have 144 observations in the dataset of whom 126 are innocent.

# ## Features

# #### Now that the dataset has been cleaned a little, I will create a few new features that will help us throughout this project.More specifically, I create 2 new features "fraction_from_poi" and "fraction_to_poi". These 2 features will record the fraction of emails a person receives from POIs and the fraction of emails he/she sends to POIs respectively. 

# In[201]:

### Task 3: Create new feature(s)-"fraction_from_poi","fraction_to_poi" recording the proportion of emails sent
### or received by POIs.
### Store to my_dataset for easy export below.
my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

def compute_fraction(poi_messages,all_messages):
    '''Returns a fraction in decimal form. Calculates proportion of all messages that were related to POIs.
    '''
    fraction=0
    if poi_messages != 'NaN' and all_messages != 'NaN':
        fraction=float(poi_messages)/float(all_messages)
    else:
        fraction=0
    return fraction

submit_dict={}
for name in my_dataset:
    data_point=my_dataset[name]
    from_poi=data_point["from_poi_to_this_person"]
    to_messages=data_point["to_messages"]
    fraction_from_poi=compute_fraction(from_poi,to_messages)
    data_point["fraction_from_poi"]=fraction_from_poi
    to_poi=data_point["from_this_person_to_poi"]
    from_messages=data_point["from_messages"]
    fraction_to_poi=compute_fraction(to_poi,from_messages)
    data_point["fraction_to_poi"]=fraction_to_poi
    submit_dict[name]={"from_poi":fraction_from_poi,
                       "to_poi":fraction_to_poi,
                        }


features_list = ["poi", "fraction_from_poi", "fraction_to_poi"]    
### store to my_dataset for easy export below
my_dataset = data_dict


### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)

### plot new features
colors=["r","b"]
for point in data:
    from_poi = point[1]
    to_poi = point[2]
    plt.scatter( from_poi, to_poi )
    if point[0] == 1:
        pl1=plt.scatter(from_poi, to_poi, color=colors[0], marker="o")
    else:
        pl2=plt.scatter(from_poi, to_poi, color= colors[1], marker="o")
plt.xlabel("fraction of emails this person gets from POIs")
plt.ylabel("fraction of emails this person sends to POIs")
plt.legend((pl1,pl2),('POI','non-POI'))
plt.title("POI vs non-POI distribution")
plt.show()


# #### From the scatterplot above, we can see that there is a little pattern to a person being a POI. Almost every POI sends 20% or more of his/her emails to other POIs and receives about 2.5-7.5% of emails from POIs.

# #### Below, I will create a list of features that I want to scrutinize further and subsequently use some of these as the basis of my classification algorithms. 

# In[202]:


my_dataset=data_dict
features_list = ['poi','salary',
                 'bonus',
                 'director_fees',
                 'exercised_stock_options',
                 'expenses',
                 'loan_advances',
                 'long_term_incentive',
                 'other',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'total_payments',
                 'total_stock_value',
                 'fraction_from_poi',
                 'fraction_to_poi',
                 'shared_receipt_with_poi']
data = featureFormat(my_dataset, features_list)
### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)

import numpy as np
import pandas as pd
#Split our features and labels into training and testing groups each. Testing size is 30% of the original data.
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#using SelectKBest to look at how influential or important each feature is.
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
selector=SelectKBest(f_classif,'all')
clf=selector.fit(features_train, labels_train)
scores=clf.scores_
scores_dict={}
for i in range (0, len(scores)):
    feature_names=features_list[i+1]
    score=scores[i]
    scores_dict[feature_names]=score
#ordering and printing scores and their respective feature names
scores_sorted=pd.DataFrame.from_dict(scores_dict,orient='index')
scores_sorted.columns=['Score']
print scores_sorted.sort_values(by=['Score'],ascending=False)


# #### Now that we have a little more idea about how powerful or influential the features are based on their scores above, we will try using various combinations of these in our algorithm classfications below. 

# ## Algorithms

# In[203]:

### Task 4: Try a varity of classifiers
### classifier named clf for easy export below.

### Gaussian Naive-Bayes 

#I am using the following features after looking at the SelectKBest scores, using the top 11 based off 
# on some manual trials involving forward selection.
features_list = ['poi','salary','bonus','exercised_stock_options','total_stock_value','expenses',
                 'long_term_incentive',
                 'fraction_from_poi','director_fees',
                 'fraction_to_poi','shared_receipt_with_poi','restricted_stock']

data = featureFormat(my_dataset, features_list, sort_keys=True)
### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)
#create the training and test sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
#Provided to give you a starting point. Try a variety of classifiers.
#Gaussian classifier
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print "Accuracy score:",accuracy_score(pred,labels_test)
print "Precision score:",precision_score(labels_test,pred)
print "Recall score:",recall_score(labels_test,pred)


# #### The Gaussian NB classifier has the following evaluation metrics on the test set:
# #### Accuracy score: 0.86 
# #### Precision score: 0.6
# #### Recall score: 0.43
# #### When tested on the entire dataset using StratifiedShuffleSplit those measurements changed a bit. Accuracy went down by about 10%. Precision went down by about 30% while recall improved by about 7% These are the measurements:
# #### Accuracy score: 0.78 
# #### Precision score: 0.30
# #### Recall score: 0.50
# #### F1 score: 0.38

# In[204]:

#Support Vector Machines(SVM)

#I am using the following features after loooking at the SelectKBest scores above
features_list = ['poi','salary','bonus','exercised_stock_options','total_stock_value','expenses',
                 'long_term_incentive',
                 'fraction_from_poi',
                 'fraction_to_poi','shared_receipt_with_poi','restricted_stock']
data = featureFormat(my_dataset, features_list, sort_keys=True)
### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)
#create the training and test sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#SVM classifier
from sklearn import svm
clf=svm.SVC()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print "Accuracy score:",accuracy_score(pred,labels_test)
print "Precision score:",precision_score(labels_test,pred)
print "Recall score:",recall_score(labels_test,pred)


# #### The SVM classifier has the following evaluation metrics on the test set:
# #### Accuracy score: 0.83 
# #### Precision score: 0.0
# #### Recall score: 0.0
# #### When tested on the entire dataset using StratifiedShuffleSplit SVM performed as badly and due to the lack of true positive predictions the metric measurements were undefined.:
# 

# In[205]:

#Decision Tree 

#After some trials involving backward selection and the DT feature importances score,I decided to use these:
features_list = [ 'poi','salary','bonus','exercised_stock_options','total_stock_value','expenses',
                 'long_term_incentive',
                 'fraction_from_poi',
                 'fraction_to_poi','shared_receipt_with_poi','restricted_stock'
                 ]
data = featureFormat(my_dataset, features_list, sort_keys=True)
### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)
#create the training and test sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#Decision Tree classifier
from sklearn import tree
from sklearn import grid_search
clf=tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print "Accuracy score:",accuracy_score(pred,labels_test)
print "Precision score:",precision_score(labels_test,pred)
print "Recall score:",recall_score(labels_test,pred)

#print clf.best_params_
imp=clf.feature_importances_
imp_dict={}
for i in range (0, len(imp)):
    feature_names=features_list[i+1]
    importance=imp[i]
    imp_dict[feature_names]=importance
#ordering and printing scores and their respective feature names
imp_sorted=pd.DataFrame.from_dict(imp_dict,orient='index')
imp_sorted.columns=['Score']
print "Feature importances for this model:"
print imp_sorted.sort_values(by=['Score'],ascending=False)


# In[206]:

#Decision Tree 

#Using backward selection and the DT feature importances score above,I decided to use these:
features_list = [ 'poi','fraction_to_poi','shared_receipt_with_poi','exercised_stock_options',
                 'long_term_incentive'
                 ]
data = featureFormat(my_dataset, features_list, sort_keys=True)
### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)
#create the training and test sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#Decision Tree classifier
from sklearn import tree
from sklearn import grid_search
clf=tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print "Accuracy score:",accuracy_score(pred,labels_test)
print "Precision score:",precision_score(labels_test,pred)
print "Recall score:",recall_score(labels_test,pred)

#print clf.best_params_
imp=clf.feature_importances_
imp_dict={}
for i in range (0, len(imp)):
    feature_names=features_list[i+1]
    importance=imp[i]
    imp_dict[feature_names]=importance
#ordering and printing scores and their respective feature names
imp_sorted=pd.DataFrame.from_dict(imp_dict,orient='index')
imp_sorted.columns=['Score']
print "Feature importances for this model:"
print imp_sorted.sort_values(by=['Score'],ascending=False)


# #### Removing the unimportant features definitely improved my algorithm's accuracy and precision. I will keep on reducing the number of features and see if that helps improve my recall score.  

# In[193]:

#Decision Tree 

#Using backward selection I remove 1 more feature from the above list and see how it changes our metrics:
features_list = [ 'poi','fraction_to_poi','shared_receipt_with_poi','exercised_stock_options'
                 ]
data = featureFormat(my_dataset, features_list, sort_keys=True)
### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)
#create the training and test sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#Decision Tree classifier
from sklearn import tree
from sklearn import grid_search
clf=tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print "Accuracy score:",accuracy_score(pred,labels_test)
print "Precision score:",precision_score(labels_test,pred)
print "Recall score:",recall_score(labels_test,pred)

#print clf.best_params_
imp=clf.feature_importances_
imp_dict={}
for i in range (0, len(imp)):
    feature_names=features_list[i+1]
    importance=imp[i]
    imp_dict[feature_names]=importance
#ordering and printing scores and their respective feature names
imp_sorted=pd.DataFrame.from_dict(imp_dict,orient='index')
imp_sorted.columns=['Score']
print "Feature importances for this model:"
print imp_sorted.sort_values(by=['Score'],ascending=False)


# #### Removing the weakest financial feature('long_term_incentive') from my previous run improved my model's recall but brought down the precision. Below, I will see how removing the weakest feature from here('exercised_stock_options') changes my results. 

# In[207]:

#Decision Tree 

#Using backward selection I remove 1 more feature from the above list and see how it changes our metrics:
features_list = [ 'poi','fraction_to_poi','shared_receipt_with_poi'
                 ]
data = featureFormat(my_dataset, features_list, sort_keys=True)
### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)
#create the training and test sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#Decision Tree classifier
from sklearn import tree
from sklearn import grid_search
clf=tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print "Accuracy score:",accuracy_score(pred,labels_test)
print "Precision score:",precision_score(labels_test,pred)
print "Recall score:",recall_score(labels_test,pred)

#print clf.best_params_
imp=clf.feature_importances_
imp_dict={}
for i in range (0, len(imp)):
    feature_names=features_list[i+1]
    importance=imp[i]
    imp_dict[feature_names]=importance
#ordering and printing scores and their respective feature names
imp_sorted=pd.DataFrame.from_dict(imp_dict,orient='index')
imp_sorted.columns=['Score']
print "Feature importances for this model:"
print imp_sorted.sort_values(by=['Score'],ascending=False)


# #### Looks like having just the email features gives us the best performance metrics. This model has the best average accuracy, precision and recall score mix we have seen so far and I am going to use these 2 features for my decision tree algorithm going forward.

# #### The Decision Tree classifier has the following evaluation metrics on the test set:
# #### Accuracy score: 0.96 
# #### Precision score: 1
# #### Recall score: 0.5
# #### When tested on the entire dataset using StratifiedShuffleSplit, accuracy went down by about 10%. Precision went down to a more realistic 34% while the recall score decreased by about 13% These are the measurements:
# #### Accuracy score: 0.85 
# #### Precision score: 0.34
# #### Recall score: 0.37
# #### F1 score: 0.35

# In[208]:

#K-Nearest Neighbors(K-NN)

#I am using the following features after looking at the SelectKBest scores above and some manual trials.
features_list = ['poi','salary','bonus','exercised_stock_options','total_stock_value','expenses',
                 'fraction_from_poi',
                 'fraction_to_poi','shared_receipt_with_poi'
                 ]
data = featureFormat(my_dataset, features_list, sort_keys=True)
### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)
#create the training and test sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#K-Nearest Neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print "Accuracy score:",accuracy_score(pred,labels_test)
print "Precision score:",precision_score(labels_test,pred)
print "Recall score:",recall_score(labels_test,pred)


# #### The K-NN classifier has the following evaluation metrics on the test set:
# #### Accuracy score: 0.90 
# #### Precision score: 0.33
# #### Recall score: 0.33
# #### When tested on the entire dataset using StratifiedShuffleSplit accuracy went down very slightly. Precision score improved by about 100%  while recall went down by about 6%. These are the measurements:
# #### Accuracy score: 0.88 
# #### Precision score: 0.65
# #### Recall score: 0.27
# #### F1 score: 0.39

# #### Now that I have run the various algorithms, I know that K-NN, Decision Tree and Guassian NB algorithms performed pretty well while the SVM didn't fare well at all. Of these, I think decision tree has the best performance because it has the best accuracy, precision and recall scores on the test set. On the entire StratifiedShuffleSplit cross validated test set, KNN does have better accuracy and precision but it has very wide apart precision and recall scores. On a dataset like this where the number of POIs are a lot less than the number of non-POIs, I expect the accuracy to be imbalanced in favor of predicting '0' or non-POI classfication. This is why I use precision, recall and f1 scores as my benchmarks while deciding my final classfier. The NB classifier does pretty well despite being so simple but it is not as good as the DT classifier. I decide to Decision tree for my final algorithm because both precision and recall scores are above 0.3 and pretty close to each other when run on the stratified test set. It also has the best results on the 30% split test set we use. I see the opportunity for some tuning here that could help me get even better performance.  

# ## Tuning and Validation

# In[209]:

###Task 5: Tuning my KNN algorithm further to achieve better than .3 precision and recall 
### using our testing script.

features_list = ['poi','salary','bonus','exercised_stock_options','total_stock_value','expenses',
                 'fraction_from_poi',
                 'fraction_to_poi','shared_receipt_with_poi'
                 ]
data = featureFormat(my_dataset, features_list, sort_keys=True)
### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import grid_search

### Split train and test sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#GridSearchCV to get a preliminary idea about the best parameters
parameters={'n_neighbors':range(1,10),'weights':['uniform','distance']}
knn=KNeighborsClassifier(algorithm='auto')
eval_scoring=['recall','precision']
for i, eval_param in enumerate(eval_scoring):
    clf=grid_search.GridSearchCV(knn,parameters,scoring=eval_param)
    clf.fit(features_train,labels_train)
    pred=clf.predict(features_test)
    print "accuracy:",accuracy_score(pred,labels_test)
    print "Precision:",precision_score(labels_test,pred)
    print "Recall:",recall_score(labels_test,pred)
    print "F1 score:",f1_score(labels_test,pred)
    print eval_param,clf.best_params_


# In[210]:

###Task 5: Tuning my choice of classifier(Decision Tree) further to achieve better than .3 precision and recall 
### using our testing script.

features_list = [ 'poi','shared_receipt_with_poi','fraction_to_poi'
                 ]
data = featureFormat(my_dataset, features_list, sort_keys=True)
### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import grid_search
from sklearn import tree
### Split train and test sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#GridSearchCV to get a preliminary idea about the best parameters
parameters={'criterion':['gini','entropy'],'min_samples_split':range(1,30),'min_samples_leaf':range(1,15)}
dt=tree.DecisionTreeClassifier()
eval_scoring=['recall','precision']
for i, eval_param in enumerate(eval_scoring):
    clf=grid_search.GridSearchCV(dt,parameters,scoring=eval_param)
    clf.fit(features_train,labels_train)
    pred=clf.predict(features_test)
    print "accuracy:",accuracy_score(pred,labels_test)
    print "Precision:",precision_score(labels_test,pred)
    print "Recall:",recall_score(labels_test,pred)
    print "F1 score:",f1_score(labels_test,pred)
    print eval_param,clf.best_params_


# #### After tuning both the Decision Tree and KNN algorithms, I want to use use Decision Tree as my final model. Below I will further tune the algorithm manually on a 10-fold cross validated train/test set such that I get a more accurate estimate of how the classifier will perform on the StratifiedShuffleSplit test set.

# In[211]:

###Task 5: Tuning my choice of classifier(Decision Tree)further to achieve better than .3 precision and recall 
### using our testing script.
features_list = [ 'poi','fraction_to_poi','shared_receipt_with_poi'
                 ]
data = featureFormat(my_dataset, features_list, sort_keys=True)
### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)
#create the training and test sets
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import grid_search
from sklearn import tree
### using a 10-fold cross-validation to see more accurate evaluation metric results.
from sklearn.cross_validation import KFold
kf=KFold(len(features),10,shuffle=True,random_state=29)
acc=0
prec=0
rec=0
f1=0
for train_indices,test_indices in kf: 
    features_train=[features[ii] for ii in train_indices]
    features_test=[features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test =[labels[ii] for ii in test_indices]
    #manually tuning the parameters to maximize accuracy, making precision and recall greater than 0.3 
    clf=tree.DecisionTreeClassifier(criterion='gini',min_samples_split=30,min_samples_leaf=8)
    clf.fit(features_train,labels_train)
    pred=clf.predict(features_test)
    acc+=accuracy_score(pred,labels_test)
    prec+=precision_score(labels_test,pred)
    rec+=recall_score(labels_test,pred)
    f1+=f1_score(labels_test,pred)
print "Average accuracy score:",acc/10.0
print "Average Precision score:",prec/10.0
print "Average Recall score:",rec/10.0
print "Average f1-score:",f1/10.0

### Task 6: Dumping my classifier, dataset, and features_list so anyone can check these results.
dump_classifier_and_data(clf, my_dataset, features_list)


# #### After tuning my Decision Tree classifier, I get pretty good results. I did have to make a tradeoff between a higher precision vs recall score. I believe the recall score is a little more important because that allows us to not ignore actual POIs. A lower precision score means that some of the people classified as POIs by this identifier will not actually be POIs in reality. I am ready to accept this downside because I feel it is better to have more POIs initially and narrow down the true criminals as the investigation progresses than to completely miss out on the guilty people right from the get go. Apart from using GridSearchCV, I manually tuned the KNN algorithm as well.I managed to get a 99% precision with it on the StratifiedShuffleSplit test set but it had a very low recall score which is why the Decision tree having an 80% recall score and about 44% precision on the stratified test set seemed the most balanced one to me. I used a 10-fold cross-validation test set above and found out the average accuracy, precision, recall and f1 scores to get a more accurate picture of how our classifier will perform on the entire stratified test set. The results we get here were indeed pretty close to how our algorithm performed on the stratifed test set.
# #### Performance on 10-fold cross validation test set:
# ##### Accuracy score: 0.85 
# ##### Precision score: 0.56
# ##### Recall score: 0.8
# ##### f1 score: 0.61
# #### On the StratifiedShuffleSplit test set:
# ##### Accuracy score: 0.86
# ##### Precision score: 0.44
# ##### Recall score: 0.80
# ##### F1 score: 0.57

# ### On a dataset like this where the number of POIs are a lot less than the number of non-POIs, I expect the accuracy to be imbalanced in favor of predicting '0' or a non-POI classification. This is why I stress more on precision and recall while evaluating the performance of my model. My identifier doesn’t have great precision, but it does have good recall. That means that, nearly every time a POI shows up in my test set, I am able to identify him or her. The cost of this is that I sometimes get some false positives, where non-POIs get flagged but as I said earlier, I chose to have better recall than precision while deciding on the tradeoff because I think it is important that the identifier doesn't miss a lot of guilty POIs from the get go from an investigative point of view. I would rather have more suspects and then narrow down as the investigation progresses than let a bunch of POIs escape just to have better precision . Having said that, our precision of 44%, although not as good as recall is not horrible. This means that of everyone classifed as a POI by the identifier, 44% actually end up as POIs in reality while 80% of all POIs in the dataset are caught by this algorithm. The F1 score, which is the harmonic average between the precision and recall scores, is 0.57. This shows that the overall performance of the algorithm is pretty decent. 
