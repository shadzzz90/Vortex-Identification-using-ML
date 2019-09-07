

import tensorflow as tf
import pandas as pd
import numpy as np
import os

tf.enable_eager_execution()



def readTestTrainData_Denselayer (features, labels, sequence_lenght = 2, split_percent = 0.8):

    df = pd.read_csv('./selected_csv.csv')

    columns = features + labels
    num_features = len(features)
    num_labels = len(labels)

    # print(df)

    df = df[columns].values

    result =[]
    # print(range(len(df)-sequence_lenght+1))
    print(len(df))
    for (index1, index2) in zip(range(0,len(df)-4,4), range(4,len(df),4)):
        print(index1,index2)

        result.append(df[index1:index2])
    result = np.array(result)
    print(result, len(result), np.shape(result))

    row = round(split_percent*result.shape[0])
    #
    train = result[:row,:,:]
    #
    # print(train, len(train), np.shape(train))
    #
    X_train = train[:,:,:-num_labels]
    y_train = train[:,:, -num_labels:]

    print(X_train, len(X_train), np.shape(X_train))
    X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], num_features))
    y_train = np.reshape(y_train,(y_train.shape[0], y_train.shape[1], num_labels))
    print(X_train, len(X_train), np.shape(X_train))
    print(y_train,len(y_train), np.shape(y_train))

    # print(df, len(df))
    return X_train, y_train


features_columns = ['VelocityX', 'VelocityY', 'VelocityZ']

label_columns = ['IVD']

X_train, y_train = readTestTrainData_Denselayer(features_columns, label_columns)


def buildTrainModel_CNN():

    input_layer = tf.keras.layers.Conv2D(16,(2,2),padding='same', input_shape=(28,28,3),activation='relu', data_format='channels_last')
    pool_layer1 = tf.keras.layers.MaxPooling2D((2,2))
    conv_layer2 = tf.keras.layers.Conv2D(16, (2,2), activation= 'relu')
    pool_layer2 = tf.keras.layers.MaxPooling2D(2,2)
    conv_layer3 = tf.keras.layers.Conv2D(64,(2,2), activation='relu')
    pool_layer3 = tf.keras.layers.MaxPooling2D(2,2)
    dropout_layer = tf.keras.layers.Dropout(0.5)
    flatten_layer = tf.keras.layers.Flatten()
    dense_layer = tf.keras.layers.Dense(512, activation= 'relu')
    output_layer = tf.keras.layers.Dense(2, activation='softmax')


    model = tf.keras.Sequential([input_layer,pool_layer1,conv_layer2,
                                 pool_layer2,conv_layer3,pool_layer3,dropout_layer,
                                 flatten_layer, dense_layer,output_layer])


    model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

#
# buildTrainModel()


def buildDenseModel():
    # layers = [X_train.shape[2], 128, 30, 20, y_train.shape[1], 32]

    l0 = tf.keras.layers.Dense(128, activation='relu', input_shape=(1,5,4,3))
    l1 = tf.keras.layers.Dense(64, activation='relu')
    l2 = tf.keras.layers.Dense(32, activation='relu')
    l3 = tf.keras.layers.Dense(16, activation='relu')
    l4  = tf.keras.layers.Dense(2, activation='softmax')

    model = tf.keras.Sequential([l0, l1, l2, l3, l4])

    model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    model.fit(X_train, y_train, epochs=10)


# buildDenseModel()


def buildDense2()



