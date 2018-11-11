from math import sqrt
import numpy
from numpy import concatenate
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from datetime import datetime
from keras.callbacks import ModelCheckpoint
import math
from conversion import forecast_to_base_format
from conversion import base_format_to_forecast
from conversion import base_second_format_to_forecast

# convert series to supervised learning
def series_to_supervised(data, n_in=0, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	#print(df)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load data
dataset = read_csv('ready_for_train_ryu.csv',header=0)
dataset.fillna(dataset.mean(),inplace=True)

values = dataset.values 
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#scaled = DataFrame(scaled)
#scaled.fillna(0,inplace=True)

scaler.inverse_transform(scaled)
full_mnt = []
print(len(scaled))
for value in scaled:
	som = 0
	for v in value[:126]:
		if not math.isnan(float(v)):
			som += float(v)
	full_mnt.append(som)
'''	
plt.plot(full_mnt)
plt.show()

exit(0)
'''
# integer encode direction
# ensure all data is float

# frame as supervised learning
n_predict_months = 12*3
reframed = series_to_supervised(scaled, n_in=0, n_out=n_predict_months)
#print(len(reframed))
# on garde seulement le temps t qu'on voudra predir
len_columns = len(dataset.columns)
#remove = [x for x in range(len_columns,(len_columns*2)-1)]
#print(remove)
#reframed.drop(reframed.columns[remove], axis=1, inplace=True)
#print(reframed.head())
values = reframed.values
#print(len_columns)
#print(reframed[:1])

# 136 - 11 = 125
# prendre jusqu'a la 125 exclure a partir de 126
# Split train test
# On veut les 2009 à 2015
n_train_months = 12*6
dataset = values[:,:]
train = values[:n_train_months,:]
test = values[n_train_months:,:]

for t in train[:1]:
	print(len(t))
train_X, train_y = train[:, :len_columns], train[:,len_columns :]
test_X, test_y = test[:, :len_columns], test[:,len_columns:]

print(len(train_X))
for t in train_X[:1]:
	print(len(t))

print(len(train_y))
for t in train_y[:1]:
	print(len(t))


train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# design network
model = Sequential()
# voir comment gérer le batch_input pour le stateful = True
#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),stateful=True))
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(LSTM(500, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
#model.add(Dense(10000))
model.add(Dense(test_y.shape[1]))
#loss='mse', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'])
model.compile(loss='mse', optimizer='sgd',metrics=['mean_squared_error'])

checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=2, save_best_only=True, mode='min')
bighistory = None
# POUR LE STATEFUL MODE
'''
for _ in range(20):
	if bighistory == None:
		bighistory = model.fit(train_X, train_y, epochs=1, batch_size=1, validation_data=(test_X, test_y),callbacks=[checkpoint] ,verbose=2, shuffle=False)
		model.reset_states()
	else:
		history = model.fit(train_X, train_y, epochs=1, batch_size=1, validation_data=(test_X, test_y),callbacks=[checkpoint] ,verbose=2, shuffle=False)
		model.reset_states()
		bighistory.history['val_loss'].append(history.history['val_loss'][0])
		bighistory.history['loss'].append(history.history['loss'][0])
'''
history = model.fit(train_X, train_y, epochs=50, batch_size=1, validation_data=(test_X, test_y),callbacks=[checkpoint] ,verbose=2, shuffle=False)

#print(history.history['val_loss'].append(history.history['val_loss']))
#history = model.fit(train_X, train_y, epochs=10, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# GET TRAINS PLOT
#print(train_X[:1])
#print(train_X[0:1])
#trainPredict= model.predict(train_X[:1])
#scaler.inverse_transform(trainPredict)

'''
full_train = []
p = 0
x = 0
som = 0
while x < len(trainPredict[0]):
	som += trainPredict[0][x]
	p+=1
	if p == 126:
		full_train.append(som)
		p = 0
		x+=11
		som=0
	else:
		x+=1
'''

trainPredict= model.predict(train_X[35:36])
print(trainPredict.shape)
trainPredict = forecast_to_base_format(len_columns,trainPredict)
trainPredict = scaler.inverse_transform(trainPredict)
trainPredict = base_second_format_to_forecast(n_predict_months,trainPredict)
'''
full_train2 = []
p = 0
x = 0
som = 0
while x < len(trainPredict[0]):
	som += trainPredict[0][x]
	p+=1
	if p == 126:
		full_train2.append(som)
		p = 0
		x+=11
		som=0
	else:
		x+=1
full_train.extend(full_train2)
'''
'''
print(len(full_train))
# GET TEST PLOT
full_test = []
len(test_X)
'''

testPredict= model.predict(test_X[0:1])
print(testPredict)
testPredict = forecast_to_base_format(len_columns,testPredict)
testPredict = scaler.inverse_transform(testPredict)
testPredict = base_second_format_to_forecast(n_predict_months,testPredict)
#scaler.inverse_transform(testPredict)
'''
print("LEN PREDICT:",len(testPredict))
som = 0
x = 0
p = 0
while x < len(testPredict[0]):
	som +=testPredict[0][x]
	p+=1
	if p == 126:
		full_test.append(som)
		p = 0
		x+=11
		som = 0
	else:
		x+=1
'''
'''
full_mnt_2D = []
for v in full_mnt:
	full_mnt_2D.append([v])

full_train_2D = []
for v in full_train:
	full_train_2D.append([v])

full_test_2D = []
for v in full_test:
	full_test_2D.append([v])
'''


full_train_2D = output_region_plot(trainPredict,"11")
full_test_2D = output_region_plot(testPredict,"11")
full_mnt_2D = output_full_plot(scaler.inverse_transform(scaled),"11")

print(full_mnt_2D)
print(full_test_2D)
print(full_train_2D)

data1PredictPlot = numpy.empty_like(full_mnt_2D)
data1PredictPlot[:,:] = numpy.nan
data1PredictPlot[0:len(full_train_2D),:] = full_train_2D

#scaler.inverse_transform(data1PredictPlot)

data2PredictPlot = numpy.empty_like(full_mnt_2D)
data2PredictPlot[:,:] = numpy.nan
data2PredictPlot[(len(full_train_2D)*2)+3:,:] = full_test_2D

#scaler.inverse_transform(data2PredictPlot)

#plt.subplot(211)
plt.plot(full_mnt_2D,label="real_data")
plt.plot(data1PredictPlot,label="train")
plt.plot(data2PredictPlot,label="test")
plt.legend()
plt.show()
'''
plt.subplot(212)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
'''
'''
plt.subplot(212)
plt.plot(history.history['mean_squared_error'], label='train')
plt.plot(history.history['val_mean_squared_error'], label='test')
plt.legend()
plt.show()
'''