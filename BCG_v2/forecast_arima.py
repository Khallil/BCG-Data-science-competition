
# Forecast and Create Dataframes

# line plot of time series
from pandas import Series,read_csv
import pandas as pd
from matplotlib import pyplot as plt
import numpy
from statsmodels.tsa.arima_model import ARIMA
from numpy import genfromtxt
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime


# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		print(dataset[i] ,dataset[i - interval])
		value = float(dataset[i]) - float(dataset[i - interval])
		diff.append(value)
	return numpy.array(diff)

# load dataset
series = read_csv('all_known.csv', header=0,usecols=[1])
series.fillna(0,inplace=True)
series = series.values

# Init Scaler
scaler = MinMaxScaler() # between 0 and 1
series.reshape(len(series))
series = scaler.fit_transform(series)

# Split into train and test set
len_full = len(series)
#split_n = int(len_full * 0.67)
#train,test = series[0:split_n],series[split_n:]
# Seasonality detection
months_in_year = 12
differenced = difference(series, months_in_year)

# Fit model
model = ARIMA(differenced, order=(7,0,1))
model_fit = model.fit(disp=0)

# Multi-step out-of-sample forecast
start_index = len(differenced)
end_index = start_index + 36
forecast = model_fit.predict(start=start_index, end=end_index)

# Invert the differenced forecast to something usable
history = [x for x in series]
month = 1
prediction = []
for yhat in forecast:
	inverted = inverse_difference(history, yhat, months_in_year)
	print('Month %d: %f' % (month, inverted))
	prediction.append(inverted)
	history.append(inverted)
	month += 1

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# evaluate forecasts with rmse et r2
#rmse = sqrt(mean_squared_error(test, prediction))
#r2 = sqrt(r2_score(test, prediction))
#mape = mean_absolute_percentage_error(test,prediction)
#print('Test RMSE: %.3f' % rmse)
#print('Test R2: %.3f' % r2)
#print('Test MAPE : %.3f' % mape)

# Inverse

# Numpy arrays
#test = numpy.asarray(scaler.inverse_transform(test)) # test
prediction = numpy.asarray(scaler.inverse_transform(prediction)) #prediction
series = scaler.inverse_transform(series) # real
# Voir comment tu peux ajouter les années et plot ça
dico = {"time":[],"dep":[]}
filled_dico = dico
years = ["2009","2010","2011","2012","2013","2014",
"2015","2016","2017"]
years_test = ["2015","2016","2017"]
years_prediction = ["2018","2019","2020"]
y = 0
i = 1
month = 1
for item in prediction:
	print("item : ",item)
	if i % 12 == 0:
		y += 1
		month = 1
	if y == 3:
		break
	if month < 10:
		date = str(years_prediction[y])+"-0"+str(month)+"-01"
	else:
		date = str(years_prediction[y])+"-"+str(month)+"-01"
	filled_dico["time"].append(date)
	filled_dico["dep"].append(item[0])
	month+=1
	i+=1


df = pd.DataFrame.from_dict(filled_dico)
df.time = df.time.apply(lambda x: datetime.strptime(x, '%Y'+'-'+'%m'+'-'+'%d'))
df.set_index('time',inplace=True)
#df.to_csv('prediction.csv')
plt.plot(df)

filled_dico = dico
y = 0
i = 1
month = 1
for item in series:
	print("item : ",item)
	if i % 12 == 0:
		y += 1
		month = 1
	if y == 3:
		break
	if month < 10:
		date = str(years[y])+"-0"+str(month)+"-01"
	else:
		date = str(years[y])+"-"+str(month)+"-01"
	filled_dico["time"].append(date)
	filled_dico["dep"].append(item[0])
	month+=1
	i+=1

df = pd.DataFrame.from_dict(filled_dico)
df.time = df.time.apply(lambda x: datetime.strptime(x, '%Y'+'-'+'%m'+'-'+'%d'))
df.set_index('time',inplace=True)
plt.plot(df)


plt.show()

'''
prediction_plot = numpy.empty_like(series)
prediction_plot[:]= numpy.nan

plt.plot(prediction_plot,label='prediction')
plt.xlabel("Evaluation Plot from 2009 to 2017")
'''
'''
series = read_csv('validation.csv', header=0,usecols=[1])
series = series.values
series = series.reshape(len(series))
'''
'''
SPLIT DATASET EN TRAIN/VAL

series = read_csv('results-final.csv', header=0)
series.fillna(0,inplace=True)

#create csvs
array = series.values
array.astype(float)
print(array.shape)
print(array)

tab = {"dep":[]}
for l in array:
	som = 0
	for v in l:
		som += v
	tab["dep"].append(som)
series = pd.DataFrame.from_dict(tab)
print(series)
# display first few rows

split_point = len(series) - 36
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')
'''

'''
#GET FULL
series = read_csv('results-final.csv', header=0)
series.fillna(0,inplace=True)

#create csvs
array = series.values
array.astype(float)
print(array.shape)
print(array)

tab = {"dep":[]}
for l in array:
	som = 0
	for v in l:
		som += v
	tab["dep"].append(som)
series = pd.DataFrame.from_dict(tab)
print("Full csv saved.")
series.to_csv('full.csv')
'''