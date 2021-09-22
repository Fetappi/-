#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tensorflow.keras.datasets import mnist 
from tensorflow import keras

# from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
# from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model


# In[2]:


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


# Add a new axis
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_train[0].shape, 'image shape')


# In[4]:


# Convert class vectors to binary class matrices.
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)


# In[5]:


# Data normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# In[6]:


# LeNet-5 model
class LeNet(Sequential):
    def __init__(self, input_shape, nb_classes):
        super().__init__()

        self.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape, padding="same"))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Flatten())
        self.add(Dense(120, activation='relu'))
        self.add(Dense(84, activation='relu'))
        self.add(Dense(nb_classes, activation='softmax'))

        self.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])


# In[7]:


model = LeNet(x_train[0].shape, num_classes)
model.load_weights('V1.h5')


# In[8]:


model.summary()


# In[9]:


# model.fit(x_train, y_train_cat, batch_size=10,
#           epochs=1, 
#           validation_split=0.2, 
#           )


# In[10]:


# Распознавание всей тестовой выборки
pred = model.predict(x_test)

predict = np.argmax(pred, axis=1)

# # Выделение неверных вариантов
mask = predict == y_test

y = y_test[~mask]
x_false = x_test[~mask]
y_false = predict[~mask]


# In[11]:


q,p=[0],[0]
def New():
    print('Истинное значение:')
    q[0]=int(input())
    print('Ложное значение:')
    p[0]=int(input())

def Numbers(y = y, y_false = y_false):
    numbers = []
    for i in range(len(y)):
        if y[i] == q[0] and y_false[i] == p[0]:
            numbers.append(i)
    return numbers


# In[12]:


# вывод конкретных неверно распознанных изображений
def Mistake(y = y, y_false = y_false, x_false = x_false):
    plt.figure(figsize=(18,10))
    n=0
    for i in range(len(y)):
        if y[i] == q[0] and y_false[i] == p[0]:
            plt.subplot(4,10,n+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x_false[i], cmap=plt.cm.binary)
            plt.xlabel("Cеть-" + str(y_false[i]) + "\n Цифра-" + str(y[i]) )
            n+=1
    plt.subplots_adjust(wspace=0, hspace=0.5)
    plt.show()


# In[14]:


def Hor(x):
    hor=[]
    for j in range(28):
        s=0.
        for k in range(28):
            s+=x[j][k]
        hor.append(s)
    return hor


# In[15]:


# сложение по горизонтали
def Horizontal(y = y, y_false = y_false):
    rang=[]
    for i in range(len(y)):
        if y[i] == q[0] and y_false[i] == p[0]:
            rang.append(Hor(x_false[i]))
    return rang


# In[16]:


def Plot(y, y_false, x_false, hor = Hor(x_false)):
    x = np.arange(0, len(hor))
    plt.figure(figsize=(18,10))
    
    plt.subplot(3,3,1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_false, cmap=plt.cm.binary)
    plt.xlabel("Cеть-" + str(y_false) + "\n Цифра-" + str(y) )
    
    plt.subplot(3,2,2)
    plt.plot(x, hor)
    plt.show()


# In[17]:


def Graf(y = y, y_false = y_false, x_false = x_false):
    n=0
    rang = Horizontal() 
    for i in range(len(y)):
        if y[i] == q[0] and y_false[i] == p[0]:
            Plot(y[i], y_false[i], x_false[i])
            n+=1


# In[18]:


# создание массива количества неверно распознаных изображений
def Gist(y = y, y_false = y_false):
    gist=[]
    for i in range(10):
        gist.append(np.zeros(10, int))

    for i in range(10):
        for j in range(len(y)):
            if i == y[j]:
                gist[i][y_false[j]]+=1

# вывод гистограммы

    x = np.arange(0, 10)
    fig, axes = plt.subplots(nrows = 2, ncols = 5,figsize=(20,10))
    i=0
    for ax in axes.flat:
        if i <= 9:
            ax.bar(x, gist[i])
            ax.set(title='%s'%i)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
            i+=1
            ax.grid()
