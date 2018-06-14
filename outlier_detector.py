from streamer import OutlierStream, StreamPCA
from pyod.models.abod import ABOD
import numpy as np

import matplotlib.pyplot as plt


class AngularBasedOutlier(OutlierStream):

    def __init__(self, inliers, outliers):
        data_total = np.concatenate((inliers, outliers), axis=0)

        OutlierStream.__init__(self, data_total, data_total)
        self.model = ABOD(n_neighbors=10,contamination=0.045)

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
