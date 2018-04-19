import numpy as np 
import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
# df_x = pd.read_csv('DTrain.csv')
# df_y = pd.read_csv('DLabel.csv')

# import data train using pandas
train = pd.read_csv('DTrain.csv')
df_x = train.values

# save label train into array
df_y = []
with open('DLabel.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
        df_y.append((', '.join(row)))

test = pd.read_csv('DTest.csv')
x_test = test.values

# for i in range(1500):
# n = i*50
# x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=1500)
# y_train.head()
# for j in range(200):
# m = j+10
rf = RandomForestClassifier(n_estimators=100, random_state=1500, max_depth=10)
rf.fit(df_x, df_y)

pred = rf.predict(x_test)

# s = y_test.values
# count = 0
# for i in range(len(pred)):
#     if pred[i] == s[i]:
#         count += 1
print (pred)
out = open('hasilrf.csv', 'w')
for i in range(len(pred)):
    out.write(pred[i] + '\n')
out.close()
# print ("==========" , "RS=" , "1500", count/len(pred) , "=============")