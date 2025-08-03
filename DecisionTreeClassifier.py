#**Information Gain**
#Entropy = Σ - pi log(pi) log to the base 2
# for two types  -0.5 log(o.5) - 0.5 log(0.5) = 1 highest possible value
#lowest entropy classification is taken

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv("bank-full.csv", delimiter=";")
df.head(10)
X = df.iloc[:,0:16]
y = df.iloc[:,16]


X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=17)


dtc = DecisionTreeClassifier()
categorised_cols = X_train.select_dtypes(include=['object'])
le = LabelEncoder()
for i in categorised_cols:
  X_train[i] = le.fit_transform(X_train[i])
  X_test[i] = le.fit_transform(X_test[i])
dtc.fit(X_train,y_train)


y_pred = dtc.predict(X_test)

print("CONFUSION MATRIX:\n")
print(confusion_matrix(y_test,y_pred),"\n")

#TP FP
#FN TN

accuracy = accuracy_score(y_test,y_pred)
print("ACCURACY SCORE : %.2f" %accuracy)
print("\nCLASSIFICATION REPORT : \n")
print(classification_report(y_test,y_pred))

print("FEATURE IMPORTANCES\n")

features = pd.DataFrame(dtc.feature_importances_, index = X.columns)
print(features.head(16))











