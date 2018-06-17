from streamer import OutlierStream, StreamPCA
from pyod.models.abod import ABOD
import numpy as np

import matplotlib.pyplot as plt


class AngularBasedOutlier(OutlierStream):

    def __init__(self, inliers, outliers):
        data_total = np.concatenate((inliers, outliers), axis=0)
        data_total = data_total[:1000]
        self.total = data_total

        OutlierStream.__init__(self, data_total, data_total)
        self.model = ABOD(n_neighbors=10,contamination=0.045)

    def train_model(self, data):
        self.model.fit(data)

    def update_model(self, data):
        return None

    def predict_model(self, data):
        return self.model.predict(data)

    def summary(self, predictions, data_stream):


        print("Outliers: {}".format(len(list(filter(lambda x: x > 0, predictions)))))
        print("Non outliers: {}".format(len(list(filter(lambda x: x <= 0, predictions)))))

        import numpy as np
        y_axes = np.linspace(0, len(predictions), len(predictions))

        xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
        Z = self.model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

        y_axes = np.linspace(0, len(self.total), len(self.total))

        x, y = zip(*self.total)
        plt.scatter(x[:10], y_axes[:10], c='red',
                    edgecolor='k', s=20)


        plt.scatter(x[10:], y_axes[10:], c='white',
                    edgecolor='k', s=20)
        #plt.scatter(y_axes,predictions)
        plt.show()
