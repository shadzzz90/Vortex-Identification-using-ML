
import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn as sk

tf.compat.v1.enable_eager_execution

df = pd.read_csv('./test_case.csv')

# print(df)

def normalize(x):
    x = (x-x.min())/(x.max()-x.min())
    return x


newDF = pd.DataFrame({'X':np.array(df['CentroidX']),'Y':np.array(df['CentroidY']),
                    'Vx': normalize(np.array(df['Velocityi'])),'Vy':normalize(np.array(df['Velocityj'])),
                          'Q': np.array(df['Q5']) })

newDF['Vx+Vy'] = newDF['Vx']+newDF['Vy']

# print(newDF)
sampleDF = pd.DataFrame(columns= ['X', 'Y', 'Vx', 'Vy' , 'Q', 'Vx+Vy'])

Vx = np.array(newDF['Vx'])
Vy = np.array(newDF['Vy'])

maxArg = []
tempDF = newDF[:0]

# for start, stop in zip(range(0,(newDF.shape[0]+4),20),range(21,(newDF.shape[0]+4)+20,20)):
# for start in range(0,len(newDF)):
#     tempDF = newDF[start:start+20]
#     maxArg.append(tempDF['Vx+Vy'].idxmax())
#
# print(maxArg, len(maxArg))

# for i, j in zip(maxArg, range(0,len(maxArg))):
#     sampleDF.loc[j] = newDF.loc[i]
#
# Q = np.array(sampleDF['Q'], dtype= int)
# sampleDF['Q'] = Q
# sampleDF.to_csv('./sample_csv.csv')








#
# df_Vy = pd.DataFrame({'CentroidX':np.array(df['CentroidX']),'CentroidY':np.array(df['CentroidY']),
#                     'VelocityY':normalize(np.array(df['Velocityj']))})


# print(newDF)

#
# # df_Vx = df_Vx.pivot_table(values='VelocityX', index='CentroidX', columns='CentroidY')
#
# df_Vx = df_Vx.sample(frac=1).reset_index(drop=True)
#
# # print(df_Vx, df_Vx2)
#
# VelocityX = np.reshape(np.array(df_Vx['VelocityX']), (5,5))
#
# # print(VelocityX, df_Vx['VelocityX'])
# df_Vy = df_Vy.sample(frac=1).reset_index(drop=True)
#
# VelocityY = np.reshape(np.array(df_Vy['VelocityY']),(5,5))
#
# # print(VelocityY)
# # Velocity_field = []
# Velocity_field = np.array([VelocityX, VelocityY])
# print(Velocity_field, Velocity_field.shape)


