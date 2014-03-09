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

nbModel = nb(xtrain, targets, None)
knnModel = knn(xtrain, targets, None)
svmModel = SVM(xtrain, targets, None)

predictList = []
cr = csv.reader(open('./predict.csv'))
for row in cr :
   newRow = []
   for i in range (0,len(row)) :
       if row[i] != None and row[i] != '' :
           valueString = row[i]   
           v = float(valueString)
       else :
           v = 0
       newRow.append(v)
   predictList.append(newRow)
predictArr = np.array(predictList)

results = nbModel.predict(predictArr)
print 'Naive-Bayes prediction = %s' % results

results = knnModel.predict(predictArr)
print 'KNN prediction = %s' % results

results = svmModel.predict(predictArr)
print 'SVM prediction = %s' % results






   

    
