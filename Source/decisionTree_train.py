# importing necessary libraries
from sklearn import datasets
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
 
# loading the iris dataset
# iris = datasets.load_iris()
 
# X -> features, y -> label
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

for i in range(100):
    # training a DescisionTreeClassifier
    from sklearn.tree import DecisionTreeClassifier
    dtree_model = DecisionTreeClassifier(max_depth = i+1).fit(X_train, y_train)
    dtree_predictions = dtree_model.predict(X_test)

    # model accuracy for X_test  
    accuracy = dtree_model.score(X_test, y_test)
    # print (dtree_predictions)
    print (i+1, "=" ,accuracy)

    # creating a confusion matrix
    # cm = confusion_matrix(y_test, dtree_predictions)