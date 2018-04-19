# importing necessary libraries
from sklearn import datasets
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# import data train using pandas
train = pd.read_csv('DTrain.csv')
X_train = train.values

# save label train into array
y_train = []
with open('DLabel.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
        y_train.append((', '.join(row)))

# open test data
test = pd.read_csv('DTest.csv')
X_test = test.values

# training a KNN classifier
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)
# Predictions
# knn_predictions = knn.predict(X_test)
out = open('hasil.csv', 'w')
for i in range(len(svm_predictions)):
    out.write(svm_predictions[i] + '\n')
out.close()
