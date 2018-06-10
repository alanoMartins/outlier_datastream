import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager

from sklearn import svm

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from streamer_outlier import EllipticEnvelopeOutlierStream

path = "dataset/covtypeNorm.csv"
data = pd.read_csv(path, header=None).iloc[:10000, :]
y_data = data.iloc[:,54]

cl1 = data.loc[data[54] == 1]
cl2 = data.loc[data[54] == 2]
cl3 = data.loc[data[54] == 3]
# cl4 = data.loc[data[54] == 4]
# cl5 = data.loc[data[54] == 5]
# cl6 = data.loc[data[54] == 6]
# cl7 = data.loc[data[54] == 7]

cl1.drop(cl1.columns[54], axis=1, inplace=True)
cl2.drop(cl2.columns[54], axis=1, inplace=True)
cl3.drop(cl3.columns[54], axis=1, inplace=True)
# cl4.drop(cl4.columns[54], axis=1, inplace=True)
# cl5.drop(cl5.columns[54], axis=1, inplace=True)
# cl6.drop(cl6.columns[54], axis=1, inplace=True)
# cl7.drop(cl7.columns[54], axis=1, inplace=True)
cls = [cl1, cl2] # cl3, cl4, cl5, cl6, cl7]

#data.drop(data.columns[54], axis=1, inplace=True)

# fit the model
new_data = np.concatenate((cl1, cl2), axis=0)

stream = EllipticEnvelopeOutlierStream(new_data)
stream.run(cl2.values)
