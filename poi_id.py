#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cluster import KMeans

import helper_functions

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Data Exploration

# Count the total number of person in the database, feature numbers and total POI

num_person = len(data_dict)
num_key = len(data_dict['ALLEN PHILLIP K'])

num_poi = 0

for t in data_dict:
    if data_dict[t]['poi'] == 1:
        num_poi += 1

print "Total number of people: " + str(num_person)
print "Total features of one entry: " + str(num_key)
print "Total number of POI: " + str(num_poi)
print "Percentage of POI in the database: " + str(1.0*num_poi/num_person)
    
    
### Task 2: Remove outliers
features_list = ['bonus','salary']
data = featureFormat(data_dict, features_list)

for point in data:
    bonus = point[0]
    salary = point[1]
    plt.scatter(bonus, salary)

plt.xlabel('Bonus')
plt.ylabel('Salary')
plt.show()

# There is one outlier in the plot which is 'TOTAL', the sum of everyone's record.
# Need to delete this outlier.

data_dict.pop('TOTAL', 0)

data = featureFormat(data_dict, features_list)

for point in data:
    bonus = point[0]
    salary = point[1]
    plt.scatter(bonus, salary)

plt.xlabel('Bonus')
plt.ylabel('Salary')
plt.show()

# Also noticed that there is one record named "THE TRAVEL AGENCY IN THE PARK" which is NOT a name. 
# Need to remove it as well.

data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# We use two ratios (poi to person and person to poi) as new features.

for i in data_dict:
    if data_dict[i]['from_poi_to_this_person'] != 'NaN':
        data_dict[i]['from_poi_to_this_person_ratio'] = 1.*data_dict[i]['from_poi_to_this_person']/data_dict[i]['to_messages']
    else:
        data_dict[i]['from_poi_to_this_person_ratio'] = 'NaN' 
     
    if data_dict[i]['from_this_person_to_poi'] != 'NaN':
        data_dict[i]['from_this_person_to_poi_ratio'] = 1.*data_dict[i]['from_this_person_to_poi']/data_dict[i]['from_messages']
    else:
        data_dict[i]['from_this_person_to_poi_ratio'] = 'NaN'

features_list  = ['poi', 'salary', 'deferral_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi','from_poi_to_this_person_ratio','from_this_person_to_poi_ratio']



data_array = featureFormat(data_dict, features_list)
poi, features = targetFeatureSplit(data_array)

# Data Split for train and test

features_train, features_test, labels_train, labels_test = train_test_split(features, poi, test_size=0.3, random_state=42)

# Feature scaling
scaler = MinMaxScaler()
rescaled_features_train = scaler.fit_transform(features_train)
rescaled_features_test = scaler.fit_transform(features_test)

# Feature selection with SelectKBest

from sklearn.feature_selection import SelectKBest 
selection = SelectKBest(k=1)

from sklearn.pipeline import Pipeline, FeatureUnion
combined_features = FeatureUnion([("univ_select", selection)])

features_transformed = selection.fit(rescaled_features_train, labels_train).transform(rescaled_features_train)

svm = SVC(kernel="linear")

pipeline = Pipeline([("features", combined_features), ("svm", svm)])

param_grid = dict(features__univ_select__k=[1, 2, 4, 6], svm__C=[1,10,1e2,1e3])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv = 5, verbose=10)
grid_search.fit(rescaled_features_train, labels_train) 
print(grid_search.best_estimator_)

# SelectKBest score
selector2 = SelectKBest()
selector2.fit(rescaled_features_train, labels_train)
scores =  selector2.scores_
print scores

### features of high importance
i = 0
N = len(scores)
features_score = []
while i < N:
    features_score.append((features_list[i+1],scores[i]))
    i = i + 1
features_score = sorted(features_score, key = lambda x: x[1],reverse = True)
print features_score

### Final feature selection
features_list = ['poi', 'exercised_stock_options', 'from_this_person_to_poi_ratio', 'expenses']


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

##############################################################################

# Naive Bayes clf

clf_NB = GaussianNB()
clf_NB.fit(features_train, labels_train)
pred_NB = clf_NB.predict(features_test)

print 'Naive Bayes: \n Accuracy: ', accuracy_score(labels_test, pred_NB)

precision = precision_score(labels_test, pred_NB)
recall = recall_score(labels_test, pred_NB)
F1_score = 2.*precision*recall/(precision + recall)

print 'Precision: ', precision
print 'Recall: ', recall
print 'F1_score: ', F1_score

##############################################################################

# AdaBoost clf

clf_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8), algorithm='SAMME.R',learning_rate=2.5, n_estimators=40)
clf_ada.fit(features_train, labels_train)
pred_ada = clf_ada.predict(features_test)


print 'AdaBoost: \n Accuracy: ', accuracy_score(labels_test, pred_ada)

precision = precision_score(labels_test, pred_ada)
recall = recall_score(labels_test, pred_ada)
F1_score = 2.*precision*recall/(precision + recall)

print 'Precision: ', precision
print 'Recall: ', recall
print 'F1_score: ', F1_score

##############################################################################

# K-means Clustering

cv = StratifiedShuffleSplit(labels, n_iter=1000)
parameters = {'n_clusters': [2], 'n_init' : [10, 20, 30], 'tol' : [10 ** (-x) for x in range(2, 6)]}
clf_k = KMeans()
clf_k = GridSearchCV(clf_k, parameters, scoring = helper_functions.scoring, cv = cv)
clf_k.fit(features, labels)
print clf_k.best_estimator_
print clf_k.best_score_
clf_k = clf_k.best_estimator_

clf_k = KMeans(n_clusters=2, tol=0.001)
clf_k.fit(features_train, labels_train)
pred_k = clf_k.predict(features_test)

print 'K-means clustering: \n Accuracy: ', accuracy_score(labels_test, pred_k)

precision = precision_score(labels_test, pred_k)
recall = recall_score(labels_test, pred_k)
F1_score = 2.*precision*recall/(precision + recall)

print 'Precision: ', precision
print 'Recall: ', recall
print 'F1_score: ', F1_score

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8), algorithm='SAMME.R',learning_rate=2.5, n_estimators=40)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)