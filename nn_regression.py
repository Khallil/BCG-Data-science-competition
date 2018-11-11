# Time series prediction with Multilayer Perceptron NN - Regression

# si on augmente la fenetre du temps
# il faut augmenter la taille du reseau egalement
# deeper and larger with more epochs too

from pandas import read_csv
import numpy
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler


dataframe = read_csv('ready_for_train_ryu.csv', engine='python',
    skipfooter=3)
dataframe.fillna(dataframe.mean(),inplace=True)
dataset = dataframe.values
dataset = dataset.astype('float32')

# Scale values with MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = StandardScaler()
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]

def create_dataset(dataset, look_back=1):
    dataX, dataY = [],[]
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back),0] # 0 pour garder les item sur la meme feature
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
    return numpy.array(dataX),numpy.array(dataY)

#plt.plot(dataset)
#plt.show(),,d,d,fyoutube
look_back = 1
trainX, trainY = create_dataset(train,look_back)
testX, testY = create_dataset(test,look_back)

model = Sequential()
model.add(Dense(1000, input_dim = look_back,activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(trainX,trainY,epochs=100,batch_size=2,validation_data=(testX,testY),verbose=2)

trainScore = model.evaluate(trainX,trainY,verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))

testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
#print(trainPredict)
#print(testPredict)
# shift predictions for plotting
# empty_like, cree une nouvelle array de la meme forme et meme type de variable
# mais avec des valeurs randoms
trainPredictPlot = numpy.empty_like(dataset)
# numpy nan set toutes les valeurs de l'array avec nan
trainPredictPlot[:, :] = numpy.nan
# de 1 a 94  = remplir avec les valeurs de trainPredict
# on decale le remplissage de l'array de la taille de look_back pour l'affichage du plot
#trainPredictPlot[0:len(trainPredict), :] = trainPredict
trainPredictPlot[0:len(trainPredict), :] = trainPredict
#print(testPredict)
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
# de 97 a 143 = remplit avec testPredict
# pareil ici aussi on decale
testPredictPlot[len(trainPredict)+1:len(dataset)-1,:] = testPredict
#print(trainPredictPlot)
#print(testPredictPlot)
#plt.plot(testPredict)
#testPredictPlot[len(trainPredict):len(dataset),:] = testPredict
#print(testPredictPlot)
#plt.plot(dataset)
#plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()