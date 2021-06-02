# author: Bojun Yin,Ziye Wang
# contact: Ziye Wang (Email: ziyewang@cug.edu.cn),

import os
from sklearn import metrics
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, SimpleRNN, Activation, BatchNormalization, Dense, LSTM, Conv1D, MaxPool1D, Flatten,GRU
# from common_func import loss_history,evaluate_method,read_data
import evaluate_method
import read_data
import csv
from keras import optimizers
from tensorflow import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#use CPU -1, GPU 0
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
random.set_seed(6)
np.random.seed(6)

#split train_test data automatically
all_data_formodel = pd.read_csv("training_data.csv")#input training data
train_set, test_set = train_test_split(all_data_formodel, test_size=0.1, random_state=46)#define the training and validation data
n_x = all_data_formodel.shape[1]
train_x = np.array(train_set.iloc[:,0:n_x-1])
train_y_1D = np.array(train_set.iloc[:, -1:])
test_x = np.array(test_set.iloc[:,0:n_x-1])
test_y_1D = np.array(test_set.iloc[:, -1:])

train_y = np_utils.to_categorical(train_y_1D, 2)
test_y = np_utils.to_categorical(test_y_1D, 2)
all_x, all_y = read_data.read_data('test_data.csv')#input test data
train_x = np.expand_dims(train_x, axis=2)
#print(train_x)
test_x = np.expand_dims(test_x, axis=2)
all_x = np.expand_dims(all_x, axis=2)

model = Sequential()
model.add(LSTM(110, batch_input_shape=(None, 5, 1), unroll=True,activation="relu"))#parameter: hidden units
model.add(Dense(2))
model.add(Activation('softmax'))
optimizer = optimizers.Adam()
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# Fit the model
history = model.fit(train_x, train_y, validation_data=(test_x, test_y),verbose=0, batch_size=7, epochs=250)#parameter: batch size and epochs
def get_all_prob_for_raster(all_x):
    y_prob_test = model.predict(all_x)  # output predict probability
    print(y_prob_test)
    y_probability_first_all = [prob[1] for prob in y_prob_test]
    #print(y_probability_first)
    with open("probability.csv", "w", newline='', encoding='utf-8') as file:#save the predict probability
        writer = csv.writer(file, delimiter=',')
        for i in y_probability_first_all:
            writer.writerow([i])
    return y_probability_first_all
#output probability
y_probability_first_all = get_all_prob_for_raster(all_x)
def get_prob_for_val(test_x):
    y_prob_test = model.predict(test_x)  # output predict probability
    #print(y_prob_test)
    y_probability_first = [prob[1] for prob in y_prob_test]
    return y_probability_first
y_probability_first = get_prob_for_val(test_x)
map_value=np.array(y_probability_first_all)
map = map_value.reshape((XX, YY))#define the rows and columns of the probability map
acc = evaluate_method.get_acc(test_y_1D, y_probability_first)  # AUC value
test_auc = metrics.roc_auc_score(all_y, y_probability_first_all)
kappa = evaluate_method.get_kappa(test_y_1D, y_probability_first)
IOA = evaluate_method.get_IOA(test_y_1D, y_probability_first)
MCC = evaluate_method.get_mcc(test_y_1D, y_probability_first)
recall = evaluate_method.get_recall(test_y_1D, y_probability_first)
precision = evaluate_method.get_precision(test_y_1D, y_probability_first)
f1 = evaluate_method.get_f1(test_y_1D, y_probability_first)
MAPE = evaluate_method.get_MAPE(test_y_1D,y_probability_first)

evaluate_method.get_ROC(test_y_1D,y_probability_first,save_path='results/roc_lstm_11.txt')
print("ACC = " + str(acc))
print("AUC = " + str(test_auc))
print(' kappa = ' + str(kappa))
print("IOA = " + str(IOA))
print("MCC = " + str(MCC))
print(' precision = ' + str(precision))
print("recall = " + str(recall))
print("f1 = " + str(f1))

acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Accuracy and Loss')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, loss, 'blue', label='Training loss')
plt.legend()
plt.show()

plt.title('Validation accuracy and Loss')
plt.plot(epochs, val_acc, 'red', label='Validation acc')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')
plt.legend()
plt.show()

#show the probability map
plt.imshow(map)