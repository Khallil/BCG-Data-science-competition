from math import sqrt
import numpy
from numpy import concatenate
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint
from keras import losses
import math
from conversion import forecast_to_base_format
from conversion import base_format_to_forecast
from conversion import output_region_plot
from conversion import output_full_plot

# NOT USED 

# convert series to supervised learning
def series_to_supervised(df, n_in=0, n_out=1, dropnan=True):
	n_vars = len(df.columns)
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
		#agg.fillna(0,inplace=True)
		agg.dropna(inplace=True)
	return agg

# read as Pandas
dataset = read_csv('ready_for_train_ryu.csv', engine='python')
len_columns = len(dataset.columns)
# Set Nan values with mean of columns
dataset.fillna(dataset.mean(),inplace=True)

# Convert into Numpys
values = dataset.values

# Scale values with MinMaxScaler
#scaler = MinMaxScaler(feature_range=(0, 1))
scaler = StandardScaler()
values = scaler.fit_transform(values)

# Get dataset values for future plot
dataset_plot = []
values_for_plot = scaler.inverse_transform(values)
for value in values_for_plot:
	som = 0
	for v in value[0:len_columns-11]:
		som += float(v)
	dataset_plot.append(som)
dataset_plot_2D = []
for d in dataset_plot:
	dataset_plot_2D.append([d])
plt.plot(dataset_plot_2D)

# Initialize Parameters
n_train = 12*6 #Train/Test split
n_lagstep = 6
n_forecast = 1 #how many timesteps you want to generate ?

# Split into Train and Test Set
train = values[:n_train,:]
test = values[n_train:,:]

# Convert numpy to Pandas
train_df = DataFrame(train)
test_df = DataFrame(test)

# Add the supervised forecast
reframed_train = series_to_supervised(train_df, n_in=n_lagstep, n_out=n_forecast+1)
reframed_test = series_to_supervised(test_df, n_in=n_lagstep, n_out=n_forecast+1)

# Convert Pandas Dataframes into Numpy arrays
train = reframed_train.values
test = reframed_test.values

# Create TrainX/Y TestX/Y
train_X, train_y = train[:,:len_columns*(n_lagstep+1)],train[:,len_columns*(n_lagstep+1):]
test_X, test_y = test[:,:len_columns*(n_lagstep+1)],test[:,len_columns*(n_lagstep+1):]

# DISPLAY TRAIN Y

# Reshape before training, why ? I don't know
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# Reshape before training, why ? I don't know
#train_X = train_X.reshape((train_X.shape[0],train_X.shape[1], 1))
#test_X = test_X.reshape((test_X.shape[0],test_X.shape[1], 1))

#batch_size = 1
model = Sequential()
#batch_size = 1
#model.add(LSTM(100, batch_input_shape=(batch_size,train_X.shape[1], 1), stateful=True))
#model.add(LSTM(1000, batch_input_shape=(batch_size,train_X.shape[1], 1), stateful=True,return_sequences=True))
#model.add(LSTM(500, batch_input_shape=(batch_size,train_X.shape[1], 1), stateful=True))
#model.add(LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(Dense(2000, input_shape=(train_X.shape[1], train_X.shape[2]),activation='relu'))
#model.add(Dense(1000 activation='relu'))

#model.add(LSTM(500, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
model.add(LSTM(400, input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(test_y.shape[1]))
model.compile(loss=losses.mean_squared_error, optimizer='sgd',metrics=['mean_squared_error'])
# Train model
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=2, save_best_only=True, mode='min')
history = model.fit(train_X, train_y, epochs=10000, batch_size=1, validation_data=(test_X, test_y),callbacks=[checkpoint] ,verbose=2, shuffle=False)
#for i in range(100000):
#	model.fit(train_X,train_y,epochs=1,batch_size=batch_size,validation_data=(test_X, test_y),verbose=2,shuffle=False)
#	print("iteration : ",i)
#	model.reset_statesc()#
#	model.save('my_model.h5')
# Run prediction and get results
train_predicted = model.predict(train_X)

#model.reset_states()
test_predicted = model.predict(test_X)

plt.subplot(211)
train_y = forecast_to_base_format(len_columns,train_y)
train_y = scaler.inverse_transform(train_y)
train_y = base_format_to_forecast(n_forecast,train_y)
t = n_lagstep
for train in train_y:
	x = 0
	train_set = []
	while(x < len_columns*(n_forecast)):
		som = 0
		for v in train[x+1:x+(len_columns-11)+1]:
			som += float(v)
		train_set.append(som)
		x+=len_columns
	# create the 2D array and add it in the plot
	train_set_2D = []
	for v in train_set:
		train_set_2D.append([v])
	train_plot = numpy.empty_like(dataset_plot_2D)
	train_plot[:,:] = numpy.nan
	train_plot[t:len(train_set_2D)+t,:] = train_set_2D
	t+=1
	#plt.plot(train_plot)

train_predicted = forecast_to_base_format(len_columns,train_predicted)
train_predicted = scaler.inverse_transform(train_predicted)
train_predicted = base_format_to_forecast(n_forecast,train_predicted)

t = n_lagstep
for train in train_predicted:
	p = 1
	x = 0
	train_set = []
	while(x < len_columns*(n_forecast)):
		som = 0
		for v in train[x+1:x+(len_columns-11)+1]:
			som += float(v)
		train_set.append(som)
		p+=1
		x+=len_columns
	# create the 2D array and add it in the plot
	train_set_2D = []
	for v in train_set:
		train_set_2D.append([v])
	train_plot = numpy.empty_like(dataset_plot_2D)
	train_plot[:,:] = numpy.nan
	train_plot[t:len(train_set_2D)+t,:] = train_set_2D
	t+=1
	plt.plot(train_plot,color='b')

p = t
test_y = forecast_to_base_format(len_columns,test_y)
test_y = scaler.inverse_transform(test_y)
test_y = base_format_to_forecast(n_forecast,test_y)
for test in test_y:
	x = 0
	test_set = []
	while(x < len_columns*(n_forecast)):
		som = 0
		for v in test[x+1:x+(len_columns-11)+1]:
			#print("first",v)
			som += float(v)
		test_set.append(som)
		x+=len_columns
	test_set_2D = []
	for v in test_set:
		test_set_2D.append([v])
	test_plot = numpy.empty_like(dataset_plot_2D)
	print(test_plot.shape)
	test_plot[:,:] = numpy.nan
	start_step = (len(train_set_2D)+t)
	stop_step = start_step+len(test_set_2D)
	#print(start_step)
	#print(stop_step)
	test_plot[start_step:stop_step,:] = test_set_2D
	#plt.plot(test_plot,color='r')
	t+=1

t = p
test_predicted = forecast_to_base_format(len_columns,test_predicted)
test_predicted = scaler.inverse_transform(test_predicted)
test_predicted = base_format_to_forecast(n_forecast,test_predicted)
print(t)
print(test_y[0][40:50])
print(test_y[0][300:310])
#print(test_y[1][:10])
print(test_predicted[0][40:50])
print(test_predicted[0][300:310])
#print(test_predicted[1][:10])
for test in test_predicted:
	x = 0
	test_set = []
	while(x < len_columns*(n_forecast)):
		som = 0
		for v in test[x+1:x+(len_columns-11)+1]:
			#print("second",v)
			som += float(v)
		#print("second y : ",som*2.5)
		test_set.append(som)
		x+=len_columns
	test_set_2D = []
	for v in test_set:
		test_set_2D.append([v])
	test_plot_p = numpy.empty_like(dataset_plot_2D)
	test_plot_p[:,:] = numpy.nan
	start_step = (len(train_set_2D)+t)
	stop_step = start_step+len(test_set_2D)
	#print(start_step)
	#print(stop_step)
	test_plot_p[start_step:stop_step,:] = test_set_2D
	plt.plot(test_plot_p,color='c')
	t+=1

'''
train_predicted = forecast_to_base_format(len_columns,train_predicted)
train_predicted = scaler.inverse_transform(train_predicted)
train_predicted = base_format_to_forecast(n_forecast,train_predicted)

train_y = forecast_to_base_format(len_columns,train_y)
train_y = scaler.inverse_transform(train_y)
train_y = base_format_to_forecast(n_forecast,train_y)

test_predicted = forecast_to_base_format(len_columns,test_predicted)
test_predicted = scaler.inverse_transform(test_predicted)
test_predicted = base_format_to_forecast(n_forecast,test_predicted)

full_train_2D = output_region_plot(train_predicted,"32",len_columns)
full_test_2D = output_region_plot(test_predicted,"32",len_columns)
full_mnt_2D = output_full_plot(scaler.inverse_transform(values),"32")

data1PredictPlot = numpy.empty_like(full_mnt_2D)
data1PredictPlot[:,:] = numpy.nan
data1PredictPlot[0:len(full_train_2D),:] = full_train_2D
data2PredictPlot = numpy.empty_like(full_mnt_2D)
data2PredictPlot[:,:] = numpy.nan
data2PredictPlot[(len(full_train_2D)*2)+3:,:] = full_test_2D
'''
#scaler.inverse_transform(data1PredictPlot)

plt.subplot(212)
plt.plot(history.history['mean_squared_error'], label='train')
plt.plot(history.history['val_mean_squared_error'], label='test')
plt.legend()
plt.show()