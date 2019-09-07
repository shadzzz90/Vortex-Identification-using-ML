
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

tf.enable_eager_execution()

def normalize(x):
    x = (x-x.min())/(x.max()-x.min())
    return x


df = pd.read_csv('./selected_csv.csv')

VelocityX = np.array(df['VelocityX'])
VelocityY = np.array(df['VelocityY'])
VelocityZ = np.array(df['VelocityZ'])


norm_VelocityX = normalize(VelocityX)
norm_VelocityY = normalize(VelocityY)
norm_VelocityZ= normalize(VelocityZ)

df['norm_VelocityX'] = norm_VelocityX
df['norm_VelocityY'] = norm_VelocityY
df['norm_VelocityZ'] = 0


X = df.iloc[:,7:10]
y = df.iloc[:,6]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

print(X_train, y_train)
print(X_test, y_test)

def buildDenseModel():
    # layers = [X_train.shape[2], 128, 30, 20, y_train.shape[1], 32]

    l0 = tf.keras.layers.Dense(128, activation='relu', input_dim=3)
    l1 = tf.keras.layers.Dense(64, activation='relu')
    l2 = tf.keras.layers.Dense(32, activation='relu')
    l3 = tf.keras.layers.Dense(16, activation='relu')
    l4  = tf.keras.layers.Dense(2, activation='softmax')

    model = tf.keras.Sequential([l0, l1, l2, l3, l4])

    model.compile(optimizer='adam', loss= tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    model.summary()

    model.fit(X_train, y_train, epochs=100)
    y_predicted = model.predict(X_test)
    # y_predicted = [x=1, if y_predicted>0.5 for x in y_predicted ]

    print(y_predicted)

buildDenseModel()