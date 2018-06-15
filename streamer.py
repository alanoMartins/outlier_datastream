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

    def run(self, data_stream, window_size=.1, slide=.05):
        self.on_start(self.data)
        self.data_stream = data_stream
        absolute_size = int(len(data_stream) * window_size)
        absolute_slide = int(absolute_size * slide)
        initial_index = 0
        final_index = absolute_size
        while final_index <= len(data_stream):
            if len(data_stream) <= final_index + absolute_slide:
                final_index = len(data_stream)
            window = data_stream[initial_index:final_index]
            self.on_receive(window, initial_index, final_index)
            initial_index += absolute_slide
            final_index += absolute_slide
        self.on_finish()


class OutlierStream(Streamer):

    def __init__(self, inliers, outliers):
        self.ground_truth = []
        [self.ground_truth.append(1) for outlier in outliers]
        [self.ground_truth.append(0) for inline in inliers]
        data_total = np.concatenate((inliers, outliers), axis=0)
        Streamer.__init__(self, data_total)
        self.predictions = np.zeros(len(data_total))

    def on_receive(self, data_window, initial_index, final_index):
        y_pred = self.predict_model(data_window)
        self.predictions[initial_index:final_index] += y_pred
        self.update_model(data_window)

    def on_finish(self):
        self.summary(self.ground_truth, self.predictions)

    def on_start(self, data):
        self.train_model(data)

    def train_model(self, data):
        raise NotImplementedError("Please Implement this method")

    def update_model(self, data):
        raise NotImplementedError("Please Implement this method")

    def predict_model(self, data):
        raise NotImplementedError("Please Implement this method")

    def summary(self, predictions):
        raise NotImplementedError("Please Implement this method")


