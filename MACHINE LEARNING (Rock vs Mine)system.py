##IMPORTING THE DEPEN#DENCIES
##LOGISTIC REGRESSION PROBLEM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
##DATA-COLLECTION AND PROCESSING

#Loading the datasets to pandas Dataframe
sonar_data=pd.read_csv('Copy of sonar data.csv',sep=',',header=None)
#Displaying the First Five rows of our data-sets
print(sonar_data.head())#This Displays the First Five rows of our Data
#To see if there is any Missing data
print(sonar_data.isnull().sum())#shows how many Missing data we have##No Missing Data  n
#Try to see how many rows and columns we have on our data
print(sonar_data.shape)#This will give us the number of rows and columns of our data
#we have 208 rows and 61 columns(208 instance of data) and (61 features of our data)
#Displaying some statistical info about our data
print(sonar_data.describe())#Describes --->Statistical Measures of the Data
#trying to see how many rock and Mines we have
print(sonar_data[60].value_counts())#how many Rocks and Mines are on our data
##Now M---->Mine and R------>Rock
print(sonar_data.groupby(60).mean())##Gives us the Mean of the Labels
#SEPERATING THE DATA AND THE LABELS
x=sonar_data.drop(columns=60,axis=0)##This gives us all the Data appart from our labels stored into one single variable
y=sonar_data[60]##TO Gives us the Target Values
##TRAINING AND TEST DATA
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=45,stratify=y)
print(x.shape,x_train.shape,x_test.shape)## enable us to see the training Data and testing data
#The targets
print(y.shape,y_train.shape,y_test.shape)
##MODEL TRAINING ------->LOGISTIC REGRESSION MODEL
Model = LogisticRegression()
#Now training the logistic regression model with training data(Learning phase of the model)
Model.fit(x_train,y_train)
#our model have already trained
##Accuracy of the training data
x_train_prediction=Model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print(training_data_accuracy)
x_test_prediction=Model.predict(x_test)##or y_pred=Model.predict(x_test)
y_pred=Model.predict(x_test)
##This is givng the Model x_test to predict y_test
testing_data_accuracy=accuracy_score(y_pred,y_test)
print(testing_data_accuracy)#Help us see the Accuracy of our model
## PREDICTIVE SYSTEM
input_data=(
0.0176,0.0172,0.0501,0.0285,0.0262,0.0351,0.0362,0.0535
    ,0.0258,0.0474,0.0526,0.1854,0.1040,0.0948,0.0912,0.1688,
0.1568,0.0375,0.1316,0.2086,0.1976,0.0946,0.1965,0.1242,0.0616,
0.2141,0.4642,0.6471,0.6340,0.6107,0.7046,0.5376,0.5934,0.8443,
0.9481,0.9705,0.7766,0.6313,0.5760,0.6148,0.5450,0.4813,0.3406,
0.1916,0.1134,0.0640,0.0911,0.0980,0.0563,0.0187,0.0088,0.0042,
0.0175,0.0171,0.0079,0.0050,0.0112,0.0179,0.0294,0.0063
)#a data from our data sets
#Mine ------> M or a Rock--------> R
#changing our input data to a Numpy array
input_data_as_numpy_array=np.asarray(input_data)#Turns our Entire input data to array
#reshaping the Numpy array as we are predicting for one instance
input_data_reshape=input_data_as_numpy_array.reshape(1,-1)
prediction=Model.predict(input_data_reshape)#This contains the prediction
if prediction=='R':
    print("The object is a Rock!")
else:
    print("The object is a Mine!")