# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 20:53:02 2021

@author: Tayyip
"""

#imports
import pandas as pd
import numpy as np
import seaborn as sbn


#Loading Data
dataFrame = pd.read_csv('vehicles.csv')

#Getting Rid of Redundant Dimensions

dataFrame = dataFrame.drop(['Unnamed: 0','id', 'url', 'region', 'region_url', 'model', 
                            'cylinders', 'title_status', 'VIN', 'drive', 'size', 'type', 
                            'paint_color', 'image_url', 'description', 'state', 'lat', 'long', 'posting_date'], axis=1)

#Dropping Tuples with Null Values
dataFrame = dataFrame.dropna()

#exDataFrame = dataFrame.copy()

#resetting indexes after dropping null values
dataFrame.reset_index(drop=True, inplace=True)

#getting unique values of unencoded attributes
brands = dataFrame['manufacturer'].unique()
trans_mission = dataFrame['transmission'].unique()
condi_tion = dataFrame['condition'].unique()
fu_el = dataFrame['fuel'].unique()

#checking is there a value that is null
summerizedData = dataFrame.isnull().sum()


#encoding string values to meaningful int values for manufacturer, transmission, fuel
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#creating label encoder
labelencoder = LabelEncoder()

#using label encoder to encode 
dataFrame['manufacturer'] = labelencoder.fit_transform(dataFrame['manufacturer'].astype(str))
manufac = {l: i for i, l in enumerate(labelencoder.classes_)}

dataFrame['transmission'] = labelencoder.fit_transform(dataFrame['transmission'].astype(str))
trans = {l: i for i, l in enumerate(labelencoder.classes_)}

#creating one hot encoder
encoder = OneHotEncoder()

#using one hot encdoer to encode condition and fuel attributes
con = dataFrame.iloc[:,3:4].values
print(con)
con[:,0] = labelencoder.fit_transform(dataFrame.iloc[:,3:4])
print(con)
con = encoder.fit_transform(con).toarray()
print(con)

C = ['excellent', 'fair', 'good', 'like new', 'new'  ,'salvage']

F = ['diesel', 'electric', 'gas', 'hybrid', 'other']

condition = pd.DataFrame(data= con, index= range(len(con)), columns=C)

fu = dataFrame.iloc[:,4:5].values
print(fu)
fu[:,0] = labelencoder.fit_transform(dataFrame.iloc[:,4:5])
print(fu)
fu = encoder.fit_transform(fu).toarray()
print(fu)

fuel = pd.DataFrame(data= fu, index=range(len(fu)), columns=F)

#adding encoded condition and fuel to dataframe 
dataFrame = dataFrame.drop(['condition', 'fuel'], axis=1)
dataFrame = pd.concat([dataFrame, condition], axis=1)
dataFrame = pd.concat([dataFrame, fuel], axis = 1)

#analysing and describing data
describedData = dataFrame.describe()

#visualizing before adjusting price
sbn.distplot(dataFrame['price'])

#deleting meaningless high prices
countThrow = int(len(dataFrame) * 0.02)
sortedDataFrame = dataFrame.sort_values('price', ascending=(False)).iloc[countThrow:]

#visualizing price density
sbn.distplot(sortedDataFrame['price'])

#deleting meaningless low prices
sortedDataFrame = sortedDataFrame[sortedDataFrame.price > 7500]

#visualizing price density
sbn.distplot(sortedDataFrame['price'])

#adjusting cars' prices before 20.000 and getting sample 0.3 
before20 = sortedDataFrame[sortedDataFrame.price < 20000]
after20 = sortedDataFrame[sortedDataFrame.price >= 20000]
before20 = before20.sample(frac = 0.3)

#concatination of before 20 and after 20
concatinated = pd.concat([after20, before20])

#visualizing last changes
sbn.distplot(concatinated['price'])

#assigning concatinated to actual dataframe
sortedDataFrame = concatinated


#visualizing count-year graph
sbn.countplot(sortedDataFrame['year'])

#calculating throw value, first and last %2 year deleting
countThrow = int(len(dataFrame) * 0.02)
sortedDataFrame = sortedDataFrame.sort_values('year', ascending=(True)).iloc[countThrow:]

#visualizing count-year graph
sbn.countplot(sortedDataFrame['year'])

groupedByYearAndMeanPrice = sortedDataFrame.groupby('year').mean()['price']

sbn.distplot(groupedByYearAndMeanPrice)

#analysing and describing data
describedDataAfterPreprop = sortedDataFrame.describe()

#understanding relation of columns
correlatedData = sortedDataFrame.corr()

priceCorrelatedData = correlatedData.iloc[:,0:1].sort_values(by='price').T

#sortedDataFrame.to_csv('Preprocessedonehot.csv', index=False)

dataFrame = sortedDataFrame


#import pickle
#with open('car_test_classifier.pkl', 'rb') as f:
#    loadedModel = pickle.load(f)
#classifier = loadedModel

#goal column 
y = dataFrame['price'].values

#remained columns
x = dataFrame.drop('price', axis=1).values

#selection of test and train data of x and y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 10)

#scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#creating model
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(64, 64, 64, 64, 64, 64, 64, 64), verbose=True)

#fitting our model
classifier.fit(x_train, y_train)

#predicting car price
prediction = classifier.predict(x_test)

#visualizing loss curve
import matplotlib.pyplot as plt
plt.plot(classifier.loss_curve_)
plt.savefig("LOSS_CURVE_.pdf")
plt.close()

#error
from sklearn.metrics import mean_absolute_error
error = mean_absolute_error(y_test, prediction)

print(error)
#saving scaler
#import joblib
#joblib.dump(scaler, 'price_scalerw.save')

#saving model
#import pickle

#with open('car_test_classifierw.pkl', 'wb') as f:
#    pickle.dump(classifier, f)
