from Event import Event
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#Reading and balancing data
file="./magic04.data"
events=[]
classCount=6688
gammaCount=0


with open(file,"r") as input:
    for x in input:
        y=x.split(",")
        event=Event(y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7],y[8],y[9],y[10])
        if event.isGamma():
            if gammaCount<classCount:
                gammaCount+=1
                events.append(event)
        else:
            events.append(event)
    input.close()
#splitting data

training,test=train_test_split(events,test_size=0.3,random_state=42)

#preparing data

train_x=[]
train_y=[]
test_x=[]
test_y=[]
for x in training:
    train_x.append(x.getArray())
    train_y.append(x.clas)
for x in test:
    test_x.append(x.getArray())
    test_y.append(x.clas)

model = keras.Sequential()
model.add(layers.Dense(12, input_dim=8, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(train_x, train_y, epochs=150, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(train_x, train_y)
print('Accuracy: %.2f' % (accuracy*100))

#Decision Tree
#decisionTree(train_x,train_y,test_x,test_y)

#Random Forest
#randomForest(train_x,train_y,test_x,test_y,n_estimators=5)

#Ada Boost
#adaBoost(train_x,train_y,test_x,test_y,n_estimators=5)

#KNN
#kNearestNeighbor(train_x,train_y,test_x,test_y,k=5)

#Naive Bayes
#naiveBayes(train_x,train_y,test_x,test_y)