#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image
#from ClassifyNB import classify
import numpy as np
import pylab as pl
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

"""
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]
"""


#########################################################
### your code goes here ###

def main():
    chris_email = [features_train[ii] for ii in range(0,len(features_train)) if labels_train[ii]==0]
    sara_email = [features_train[ii] for ii in range(0,len(features_train)) if labels_train[ii]==1]
    
    #clf = classify(features_train, labels_train)
    
    ### draw the decision boundary with the text points overlaid
   

    X = features_train
    Y = labels_train
    x = features_test
    y = labels_test
        
    ### your code goes here!
    from sklearn.naive_bayes import GaussianNB
    
    print features_train
    print labels_train    
    
    t0 = time()
    clf = GaussianNB()
    
    clf2 =  clf.fit(X, Y)
    print "Training Time: ", round(time()-t0,3),"s"
    
    t1 = time()    
    result = clf2.predict(x)
    print "Prediction Time: ", round(time()-t1,3),"s"
    accuracy = accuracy_score(result,labels_test)
    print result
    print accuracy 
    #prettyPicture(result, features_train, labels_train, features_test, labels_test)
    #output_image("test.png", "png", open("test.png", "rb").read())
    print "All good!"

#########################################################


if __name__ == "__main__": main()