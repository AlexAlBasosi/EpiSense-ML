
'''SVM Program 
Application: "EpiSense"
Author: Rsquare
'''
import numpy as np
import pandas as pd
from sklearn import svm, neighbors, preprocessing
import csv
import time 
from time import gmtime, strftime
from pandas import DataFrame 
from sklearn import cross_validation
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import re
import requests
import httplib, urllib

	
data = pd.read_csv("FINALTRAIN.csv", )  

data = data.drop("id",1)

#opening a file to write data to it


features = list(data.columns[3:22])
x = data[features]
y = data["diagnosis"]



# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.3, random_state=42)
#normalization
x_train_N = (x_train-x_train.mean())/(x_train.max()-x_train.min())
x_test_N = (x_test-x_test.mean())/(x_test.max()-x_test.min())

clf = svm.SVC()
fitme = clf.fit(x_train, y_train)
accuracy = clf.score(x_test,y_test)



#USING K-FOLD VALIDATION
'''
Kfold = KFold(len(data),n_folds=10,shuffle=False)
print("SVM accuracy after using KFold is %s" %cross_val_score(fitme,x,y,cv=10).mean())
'''

#prints the no. of rows and columns in the file
#print(data.shape) 
 
 
#To optimise the column reading
#testing with PCA
'''pca = PCA(n_components=2)
pca.fit(x_train)
x_t_train = pca.transform(x_train)
x_t_test = pca.transform(x_test)
clf = SVC()
clf.fit(x_t_train, y_train)
print 'score', clf.score(x_t_test, y_test)'''

#deployment:
#joblib.dump(ypred2, 'saveprogram.pkl' )

########### Beschier EEG Logger ###############
from classes import EmotivTasks
from classes import Helpers
from classes import Gui

emotivHeadsetTasks = EmotivTasks.EmotivHeadsetThreadedTasks()


def onMainWindowClose():
    emotivHeadsetTasks.emotiv.engineDisconnect()

w = Helpers.FullScreenWindow(onMainWindowClose)

gui = Gui.GUI(w.tk, emotivHeadsetTasks)

# To test new data
testdata = pd.read_csv("FINALTEST.csv")
testdata = testdata.drop("id",1)


testfeatures = list(testdata.columns[3:22]) 
x2 = testdata[testfeatures]

#Now test to predict the binary values for the Y(unknown, output vector)
ypred2 = clf.predict(x2)


#To display time while giving output
currenttime1 = strftime("%Y-%m-%d %H:%M:%S", gmtime())
print(ypred2)


  
#For printing the output along with the time frame
   
df = DataFrame({'Time': currenttime1, 'Output': ypred2})    

df.to_csv('FINALOUTPUT.csv')



f = open('FINALOUTPUT.csv', "r")
readCSV = csv.reader('FINALOUTPUT.csv',delimiter= ",")

timestamp1 = strftime("%Y-%m-%d %H:%M:%S", gmtime())

for col in readCSV :
              
	a="15983851284931148960382"
	m = re.search("1111111111",a)
	if a:
	
		url = 'http://localhost:3001/patients/2/history?timestamp=' + timestamp1 + '&isSeizure=1'
		query = {'isSeizure': 1}
		res = requests.post(url, data=query)
		print(res.text)
		
		
	else:
		print("sorry bruv")
			


#**********
