#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
from sklearn import cross_validation
features_train, features_test, target_train, target_test = cross_validation.train_test_split(features, labels, random_state = 42, test_size = 0.30)


### your code goes here 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

classifier = DecisionTreeClassifier()
classifier.fit(features_train, target_train)
pred = classifier.predict(features_test)
print accuracy_score(pred,target_test )

print "How many POIs are predicted for the test set for your POI identifier?"
print target_test.count(1.0)

print "How many people total are in your test set?"
print len(target_test)

print "If your identifier predicted 0. (not POI) for everyone in the test set, what would its accuracy be?"
print target_test.count(0.0)*1.0/len(target_test)
tp = 0
tn = 0
for actual, predicted in zip(target_test, pred) :
  if(actual ==1) and (predicted == 1) :
    tp = tp+1

print "true positives"
print tp

print "precision"
print precision_score(pred, target_test)


print "recall"
print recall_score(pred, target_test)