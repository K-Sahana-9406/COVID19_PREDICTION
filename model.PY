#importing
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#splitting feature and target

patient_data=pd.read_csv('DATASET/peerj-cs-09-1333-s001.csv')

patient_data=patient_data.replace({'Yes':1,'No':1})
x_data=patient_data.drop(columns='COVID-19',axis=1)
y=patient_data['COVID-19']

#Scaling

scaler=StandardScaler()
x=scaler.fit_transform(x_data)

#splitting features and target into test and training datas

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
# print(x.shape,x_train.shape,x_test.shape)
#
# OUTPUT : (5434, 20) (4347, 20) (1087, 20)

