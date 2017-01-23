#!/usr/bin/python
import random
from scipy.spatial import distance

#returns eucledian distance between two points
def euc(a,b):
	return distance.euclidean(a,b)

#user defined classifier (K nearest neighbor)
#uses eucledian distance between testing data point and nearest training data point to make prediction
class my_KNN():
	def fit(self, x_train,y_train):
		self.x_train=x_train
		self.y_train=y_train
	
	#used to make the prediction
	def predict(self, x_test):
		predictions=[]
		for row in x_test:
			label=self.closest(row)
			predictions.append(label)
		return predictions
	
	#finds the closest training data point
	def closest(self, row):
		best_dist=euc(row,self.x_train[0])
		best_index=0
		for i in range(1,len(self.x_train)):
			dist=euc(row,self.x_train[i])
			if dist<best_dist:
				best_dist=dist
				best_index=i
		return self.y_train[best_index]
	


from sklearn import datasets
#using the iris dataset to test our algorithm
iris=datasets.load_iris()

x=iris.data
y=iris.target

from sklearn.cross_validation import train_test_split
#splitting the data set into training and testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)

#using user defined classifier for prediction
my_clf=my_KNN()
my_clf.fit(x_train,y_train)
predictions=my_clf.predict(x_test)
#finding accuracy of our 
from sklearn.metrics import accuracy_score
print "K-Neighbor accuracy:",accuracy_score(y_test,predictions)
