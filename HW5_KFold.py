import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
#reading data
iris=pd.read_csv("C:/Users/shrut/Documents/MachineLearning/HW/iris.csv")
from sklearn.neighbors import KNeighborsClassifier
#creating features and labels
X= iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
from sklearn.model_selection import train_test_split
#splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6)
kf = KFold(n_splits=5,shuffle=False)
for i,(train_set,test_set)in enumerate(kf.split(X)):
	print(train_set,test_set)
#cretaing KNN clasiifier model
classifier = KNeighborsClassifier(n_neighbors=5)
trainingmodel=classifier.fit(X_train, y_train)
#saving model in .txt file
output=open("C:/Users/shrut/Documents/MachineLearning/HW/KNNmodel.txt","w")
output.write(str(trainingmodel))
output.flush()
output.close()
#predicting values based on model
y_pred = trainingmodel.predict(X_test)
from sklearn.model_selection import cross_val_score
#performing cross validation and calculating accuracy
scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
print(scores)
#saving accuracies in .txt file
output=open("C:/Users/shrut/Documents/MachineLearning/HW/accuracy_knn.txt","w")
output.write(str(scores))
output.flush()
output.close()
#average of all accuracies
print(scores.mean())
#saving average in .txt file
output=open("C:/Users/shrut/Documents/MachineLearning/HW/mean_accuracy_knn.txt","w")
output.write(str(scores.mean()))
output.flush()
output.close()
#plotting bar graph for k vs accuracies
k=(1,2,3,4,5)
plt.bar(k,scores,color='blue',align='center', alpha=0.5)
plt.xlabel('Value of K')
plt.ylabel('Cross-validated accuracy')
#saving bar graph
plt.savefig("C:/Users/shrut/Documents/MachineLearning/HW/knnaccuracy.png")
#Decisiontree creation
from sklearn.tree import DecisionTreeClassifier
decisiontree_classifier=DecisionTreeClassifier()
#training data using decision tree
decisiontreemodel=decisiontree_classifier.fit(X_train, y_train)
#saving decision tree model in .txt file
output=open("C:/Users/shrut/Documents/MachineLearning/HW/decisiontreemodel.txt","w")
output.write(str(decisiontreemodel))
output.flush()
output.close()
#predicting values based on dt model
y_pred_dt = decisiontreemodel.predict(X_test)
#cross validation and calculating accuracy
scores_decisiontree = cross_val_score(decisiontree_classifier, X, y, cv=5, scoring='accuracy')
print(scores_decisiontree)
#saving accuracies in .txt file
output=open("C:/Users/shrut/Documents/MachineLearning/HW/decisiontree_accuracy.txt","w")
output.write(str(scores_decisiontree))
output.flush()
output.close()
#average of all accuracies
print(scores_decisiontree.mean())
#saving average in .txt file
output=open("C:/Users/shrut/Documents/MachineLearning/HW/decisiontree_mean_accuracy.txt","w")
output.write(str(scores_decisiontree.mean()))
output.flush()
output.close()
#plotting bar graph for k vs accuracies
plt.bar(k,scores_decisiontree,color='green',align='center', alpha=0.5)
plt.xlabel('Value of K')
plt.ylabel('Cross-validated accuracy')
#saving bar graph 
plt.savefig("C:/Users/shrut/Documents/MachineLearning/HW/decisiontree_accuracy.png")




	
