# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:06:03 2023

@author: ACER
"""



from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

File_path = 'C:/SEAi/data/'

File_name = 'car_data.csv'

df = pd.read_csv(File_path+File_name)

df.dropna(inplace=True)

#ไม่ใช้ข้อมูลคอลลัมน์ User id----------------------------------------
df.drop(columns=['User ID'], inplace=True)
 

#ข้อมูลเป็น String -- encode every colume-----------------------
encoders = []
for i in range(0, len(df.columns) -1):
    enc = LabelEncoder()
    df.iloc[:,i] = enc.fit_transform(df.iloc[:, i])
    encoders.append(enc)

#เตรียมข้อมูลไว้ Train----------------------------------------------
x = df.iloc[:,0:3]
y = df['Purchased']

model = DecisionTreeClassifier(criterion='entropy') #'gini'
model.fit(x, y)

#เตรียมข้อมูลทดสอบ--------------------------------------------------
x_pred = ['Female',35, 43500]
 
 #Encode data----------------------------------------------
for i in range(0, len(df.columns) -1):
      x_pred[i] = encoders[i].transform([x_pred[i]])
     
 #Reshape
x_pred_adj = np.array(x_pred).reshape(-1, 3) 

#Predict and show------------------------------------------
y_pred = model.predict(x_pred_adj)
print('Prediction:', y_pred[0])
score = model.score(x, y)
print('Accuracy','{:.2f}'.format(score))
   
#plot tree -------------------------------------------------
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

#แปลงให้เป็น List
feature = x.columns.tolist()
Data_class = y.tolist()

plt.figure(figsize=(25,20))
_ = plot_tree(model,feature_names= feature, 
              class_names= Data_class,
              label='all', 
              impurity= True,
              precision= 3,
              filled= True,
              rounded= True,
              fontsize= 16)

plt.show() 

#plot graph--------------------------------------------------

import seaborn as sns
Feature_imp = model.feature_importances_
feature_names = ['Gender','Age','AnnualSalary']
 
sns.set(rc = {'figure.figsize' : (11.7,8.7)})
sns.barplot(x = Feature_imp, y = feature_names)
 
print(Feature_imp)