from itertools import _Predicate
from os import replace
from numpy.core.multiarray import array
from numpy.lib.arraysetops import unique
import pandas as pd
import numpy as np 
from pandas.core.algorithms import value_counts
from sklearn import model_selection
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import column_or_1d
cars_data=pd.read_csv('https://drive.google.com/file/d/1UIj8rOmDJn4UgqDTJezMNfzr7jh_behL/view?usp=sharing')
cars_data.head()
cars_data.dorps(columns=['torque'],inpalce=True)
cars_data.head()
cars_data.shape
cars_data.isnull().sum()
cars_data.dropna(inplace=np.True)
cars_data.shapes
cars_data.duplicated().sum()
cars_data.drop_duplicates(inplace=True)
cars_data.shapes
cars_data
cars_data.info()
for col in cars_data.columns:
  print('unique values of' + col)
  print(cars_data[col].uniquie())
  print('---------------\n')
 def get_brand_name(car_name):
   car_name=car_name.split('')[0]
  return car_name.strip()
def clean value(value):
 value=value.split('')[0]
value=value.strip()
if value=='':
  value=0
return float(value)
get_brand_name('Maruti Swift Dzire VDI')
cars_data['names']=cars_data['name'].apply(get_brand_name)
cars_data['names'].unique()
array(['Maruti','Skoda','Honda','Hyundai ','Toyota','Ford','Renault','Mahindra','Tata','Chevrolet','datsun,'jeep','Mercedes-Benz','Mitsubishi','Audi','volkswagen','BMW','nissan','Lexus','Jaguar','Land','MG','Volvo','Daewoo','Kia','Fiat','Force','Ambassdor','Ashok','Isuzu','Opel'],)
cars_data['mileage']=cars_data['mileage'].apply(get_brand_name)
cars_data['max_power']=cars_data['max_power'].apply(get_brand_name)
  cars_data['engine']=cars_data['engine'].apply(get_brand_name)
for col in cars_data.columns:
  print('unique values of' + col)
  print(cars_data[col].uniquie())
  print('---------------\n')
  cars_data['name'].replace(['Maruti','Skoda','Honda','Hyundai ','Toyota','Ford','Renault','Mahindra','Tata','Chevrolet','datsun,'jeep','Mercedes-Benz','Mitsubishi','Audi','volkswagen','BMW','nissan','Lexus','Jaguar','Land','MG','Volvo','Daewoo','Kia','Fiat','Force','Ambassdor','Ashok','Isuzu','Opel'],
[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],inplace=True)
  cars_data['tarnsmission'].unique()
  array(['Manual','automatic'],dtypye=object)
  cars_data['transmission'].replace(['Manual','Automatic'],[1,2],inplace=True)
  cars_data['seller_type'].unique()
 array(['Individual','Dealer','Trustmarks Dealer'],dtypye=object)
cars_data['seller_type'].replace(['Individual','Dealer','Trustmarks Dealer'],[1,2,3],inplace=True
cars_data['fuel'].unique()
cars_data['fuel'].replace(['Diesel','Petrol','LPG','CNG'],[1,2,3,4],inplace=True
cars_data.info()
cars_data.reset_index(inplace=True)
cars_data
cars_data['owner'].unique()
array('First owner','Second owner','Third owner','Frouth & above owner','Test Drive car'),
cars_data['owner'].replace(['First owner','Second owner','Third owner','Frouth & above owner','Test Drive car'],[1,2,3,4,5],inplace=True)
 cars_data.drop(column=['index'],inplace=True)
cars_data
input_data=cars_data.drop(column=['selling_price'])
output_data=cars_data['selling_price']
x_train,x_test,y_train,y_test -train_test_split(input_data ,output_data,test_size=0.2)
#model Creation
model= LinearRegression()
#Train model
model.fit(x_train,y_train)
Predict=model._Predict(x_test)
_Predicate
x_train.head(1)
input_data_model= pd.DataFrames(
  [[5,2014,120000,1,1,1,1,12.99,2494.0,100.6,8.0]]
columns=['name','year','km_driven','seller_type','transmission','owner','owner','engine','max_power','seats'])
input_data_models
  model.predict(input_data_models)
  
   