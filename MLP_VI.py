


import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)  # Turnoff the tensorflow warnings

pd.set_option('display.max_columns', None)  # Display all columns
# pd.set_option('display.max_rows', None)

# Read the Data file

df = pd.read_csv('./sample.csv')

SAMPLE_SIZE = int(0.8*len(df))

# Drop the columns from the dataframe which are not required

data = df.drop(['Velocity:0','Velocity:1','Velocity:2','Vorticity:0', 'Vorticity:1','Vorticity:2',
                'Cell Type', 'vtkOriginalIndices', '__vtkIsSelected__', 'vtkCompositeIndexArray'], axis=1)


# Rearrange the columns

Q5 = np.array(data['Q5'])
data = data.drop('Q5', axis= 1)
data['Q5'] = Q5
# print(data.describe())

features = ['Pressure', 'TurbulentKineticEnergy', 'TurbulentViscosity', 'Velocityi',  'Velocityj', 'Vorticityk']
target = ['Q5']

# Using Feature Importance to select the feature.

X = data.loc[:, features] # feature
Y = np.ravel(data.loc[:, target])   # label
model = ExtraTreesClassifier()
model.fit(X,Y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(6).plot(kind = 'barh')
plt.show()


# Create the heatmap to indentify the features.

# data = data.drop(['CentroidX', 'CentroidY', 'CentroidZ', 'Velocityk', 'Vorticityi', 'Vorticityj'], axis=1)
# print(data.columns)
# corrmat = data.corr()      # getting correlation matrix
# top_corr_features = corrmat.index
# print(top_corr_features)
# plt.figure(figsize = (10,10))
#
# g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
#
# plt.show()


# Using PCA to drop down to two features.

def convertPCA (dataframe, features, targets):

    X_values = dataframe.loc[:, features].values
    # Y_values = dataframe.loc[:, target].values

    X_values_scaled = StandardScaler().fit_transform(X_values) # using standard scaler

    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(X_values_scaled)

    principalDf = pd.DataFrame(data= principalComponents, columns= ['PCA1', 'PCA2'])

    finalDf = pd.concat([dataframe.iloc[:,0:2],principalDf, dataframe[targets]], axis=1)

    return finalDf


def Shuffle_RandomSample_Dataset (dataframe):

    dataframe =shuffle(dataframe)
    dataframe = dataframe.sample(SAMPLE_SIZE,replace=False)
    dataframe.reset_index(inplace=True, drop=True)

    return dataframe


finalDf = convertPCA(data, features, target)
SampledDf = Shuffle_RandomSample_Dataset(finalDf)
SampledDf.to_csv('./randomsampled2.csv')
# print(finalDf,len(finalDf), SampledDf, len(SampledDf))

# Split training and test data

train_data, val_data = train_test_split(SampledDf, train_size=0.7, random_state= 10)

print(train_data.describe())


