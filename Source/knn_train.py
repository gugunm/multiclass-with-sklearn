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
X = train.values

y = []
with open('DLabel.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
        y.append((', '.join(row)))

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

for i in range(100):
    # training a KNN classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = i+1).fit(X_train, y_train)
    
    # accuracy on X_test
    accuracy = knn.score(X_test, y_test)
    
    # creating a confusion matrix
    knn_predictions = knn.predict(X_test) 
    cm = confusion_matrix(y_test, knn_predictions)

    print (i+1, "=" ,accuracy*100,'%')