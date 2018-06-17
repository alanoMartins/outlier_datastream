from streamer import OutlierStream, StreamPCA
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt


class EllipticEnvelopeOutlierStream(OutlierStream):

    def __init__(self, data, data_stream):
        OutlierStream.__init__(self, data, data_stream)
        self.model = EllipticEnvelope(contamination=0.045)
        self.DEBUG = False
        self.pca_plot = StreamPCA()

    def train_model(self, data):
        self.model.fit(data)

    def update_model(self, data):
        return None

    def predict_model(self, data):
        return self.model.predict(data)

    def summary(self, predictions, data_stream):

        print("Non outliers: {}".format(len(list(filter(lambda x: x > 0, predictions)))))
        print("Outliers: {}".format(len(list(filter(lambda x: x < 0, predictions)))))

        import numpy as np
        y_axes = np.linspace(0, len(predictions), len(predictions))
        plt.scatter(y_axes,predictions)
        plt.show()
