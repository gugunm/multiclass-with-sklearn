# importing necessary libraries
from sklearn import datasets
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

'''
# loading the iris dataset
iris = datasets.load_iris() 
# X -> features, y -> label
X = iris.data
y = iris.target
'''

train = pd.read_csv('DTrain.csv')
# y  = pd.read_csv('DLabel.csv')
X = train.values
# print (test)

y = []
with open('DLabel.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
        y.append((', '.join(row)))


# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
 
# training a linear SVM classifier
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)
# print (iris)
# print (X)
# print (y)
# print (svm_predictions)

# model accuracy for X_test  
accuracy = svm_model_linear.score(X_test, y_test)
# print (svm_predictions)
print (accuracy)

# creating a confusion matrix
# cm = confusion_matrix(y_test, svm_predictions)
# print (cm)'''