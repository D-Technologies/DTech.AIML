import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import csv

f = open('main.csv','w')
writer=csv.writer(f)
writer.writerow(["Price","Volume(in_millon)"])
writer.writerow([17188.65,5.37])
writer.writerow([17235.55,5.18])
writer.writerow([17237.00,5.49])
writer.writerow([17334.65,4.66])
f.close()

df=pd.read_csv('main.csv')

print("NIFTY MARKET PREDICTION")

print(df)

x = np.array(df["Price"]).reshape(-1,1)
y = np.array(df["Volume(in_millon)"]).reshape(-1,1)

x_train,x_test,y_train,y_test=train_test_split(x,y)

clf=LinearRegression()

fitting = clf.fit(x_train,y_train)

print(fitting)

price_predict = clf.predict(y_test)
volume_predict = clf.predict(x_test)

print("The price will be " + str(price_predict))
print("The volume will be " + str(volume_predict))

