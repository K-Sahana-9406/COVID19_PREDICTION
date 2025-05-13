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

patient_data=patient_data.replace({'Yes':1,'No':0})
x_data=patient_data.drop(columns='COVID-19',axis=1)
y=patient_data['COVID-19']

#Scaling

scaler=StandardScaler()
x=scaler.fit_transform(x_data)

#splitting features and target into test and training datas

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
# print(x.shape,x_train.shape,x_test.shape)
# OUTPUT : (5434, 20) (4347, 20) (1087, 20)
#Model Training

model=LogisticRegression()
model.fit(x_train,y_train)

#evaluating Model
x_train_prediction=model.predict(x_train)
# training_data_accuracy=accuracy_score(x_train_prediction,y_train)
# print("training data accuracy :",training_data_accuracy)
x_test_prediction=model.predict(x_test)
# test_data_accuracy=accuracy_score(x_test_prediction,y_test)
# print("test data accuracy :",test_data_accuracy)
with open('model.pkl','wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl','wb') as f:
    pickle.dump(scaler, f)


