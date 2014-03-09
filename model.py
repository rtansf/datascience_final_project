import csv
import numpy as np
from classifiers import *

trainingList = []
cr = csv.reader(open('./xtrain.csv'))
for row in cr :
   newRow = []
   for i in range (0,len(row)) :
       if row[i] != None and row[i] != '' :
           valueString = row[i]   
           v = float(valueString)
       else :
           v = 0
       newRow.append(v)
   trainingList.append(newRow)
xtrain = np.array(trainingList)

targetList = []
cr = csv.reader(open('./labels.csv'))
for row in cr:
    targetList.append(int(row[0]))
targets = np.array(targetList)

targetNamesList = []
cr = csv.reader(open('./label_names.csv'))
for row in cr:
    targetNamesList.append(row)
targetNames = np.array(targetNamesList)

numFolds = 10

# KNN
cv_a = cross_validate(xtrain, targets, knn, numFolds, None)
print 'KNN %s folds, score = %s' % (numFolds, cv_a)
 
# Naive Bayes
cv_a = cross_validate(xtrain, targets, nb, numFolds, None)
print 'Naive-Bayes %s folds, score = %s' % (numFolds, cv_a)

# SVM
cv_a = cross_validate(xtrain, targets, SVM, numFolds, None)
print 'SVM %s folds, score = %s' % (numFolds, cv_a)




   

    
