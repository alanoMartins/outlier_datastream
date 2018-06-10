import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
import numpy as np


class StreamerPlot:

    def on_start(self, data, border_function=None):
        self.data = data
        ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
        x, y = zip(*data)
        if border_function is not None:
            self.customize(border_function)
        self.scatter = plt.scatter(x, y, c='white',
                edgecolor='k', s=20)
        plt.ion()
        plt.show()

    def customize(self, function):
        xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
        Z = function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.title("Outlier detection")
        plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

    def on_receive(self, data, is_outlier):
        print("Time init")
        x, y = zip(data)
        if is_outlier == -1:
            plt.scatter(x, y, c='red', edgecolor='k', s=20)
        else:
            plt.scatter(x, y, c='green', edgecolor='k', s=20)
        plt.draw()
        time.sleep(0.1)
        plt.pause(0.0001)
        print("Time end")

    def on_finishing(self):
        print("Flush data")


class StreamPCA:

    def __init__(self):
        self.transform_pca = PCA(n_components=2)
        self.plot = StreamerPlot()

    def on_start(self, data, model):
        train_data = self.transform_pca.fit_transform(data)
        self.pca_model = model.fit(train_data)
        self.plot.on_start(train_data, self.pca_model.decision_function)

    def on_receive(self, data, y_pred):
        pca_data = self.transform_pca.fit_transform(data)
        y_pred1 = self.pca_model.predict(pca_data)
        for idx in range(0, len(pca_data)):
            self.plot.on_receive(pca_data[idx], y_pred[idx])
        self.plot.on_finishing()


class Streamer:

    def on_start(self, data):
        raise NotImplementedError("Please Implement this method")

    def on_receive(self, data):
        raise NotImplementedError("Please Implement this method")

    def on_finish(self):
        raise NotImplementedError("Please Implement this method")

    def __init__(self, data):
        self.data = data

    def run(self, data_stream):
        self.on_start(self.data)
        for d in data_stream:
            self.on_receive(d)
        self.on_finish()


class OutlierStream(Streamer):

    def __init__(self, data):
        Streamer.__init__(self, data)
        self.predictions = []
        self.data_stream = []

    def on_receive(self, data):
        y_pred = self.predict_model(data)
        self.predictions.append(y_pred)
        self.data_stream.append(data)
        self.update_model(data)

    def on_finish(self):
        self.summary(self.predictions, self.data_stream)

    def on_start(self, data):
        self.train_model(data)

    def train_model(self, data):
        raise NotImplementedError("Please Implement this method")

    def update_model(self, data):
        raise NotImplementedError("Please Implement this method")

    def predict_model(self, data):
        raise NotImplementedError("Please Implement this method")

    def summary(self, predictions, data_stream):
        raise NotImplementedError("Please Implement this method")


