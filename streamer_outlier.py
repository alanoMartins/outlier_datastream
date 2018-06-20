from streamer import OutlierStream
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from scipy import stats


class TemplateOutlier(OutlierStream):

    def __init__(self, inliers, outliers, model):
        data_total = np.concatenate((inliers, outliers), axis=0)
        self.data_total = data_total
        self.outliers = outliers
        self.inliers = inliers

        OutlierStream.__init__(self, inliers, outliers)
        self.model = model

    def train_model(self, data):
        self.model.fit(data)
        scores_pred = self.model.decision_function(data) * -1
        self.threshold = stats.scoreatpercentile(scores_pred, 100 * 0.006)

    def update_model(self, data):
        return None

    def predict_model(self, data):
        return self.model.predict(data)

    def summary(self, ground_truth, predictions, is_plot=False):

        predictions = list(map(lambda x: 1 if x > 0 else 0, predictions))

        acc = accuracy_score(predictions, ground_truth)
        precision = precision_score(predictions, ground_truth)
        recall = recall_score(predictions, ground_truth)
        f1 = f1_score(predictions, ground_truth)

        self.stats["f1"] = f1
        self.stats["recall"] = recall
        self.stats["precision"] = precision
        self.stats["accuracy"] = acc

        print(confusion_matrix(predictions, ground_truth))
        print("Acuracia: {}".format(acc))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))

        if is_plot:
            self._plot()

    def _plot(self):

        xx, yy = np.meshgrid(np.linspace(-70, 70, 100), np.linspace(-70, 70, 100))
        Z = self.model.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
        Z = Z.reshape(xx.shape)

        subplot = plt.subplot(1, 1, 1)
        subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), self.threshold, 7), cmap=plt.cm.Blues_r)
        subplot.contourf(xx, yy, Z, levels=[self.threshold, Z.max()], colors='orange')

        a = subplot.contour(xx, yy, Z, levels=[self.threshold], linewidths=2, colors='red')
        subplot.contourf(xx, yy, Z, levels=[self.threshold, Z.max()], colors='orange')
        b = subplot.scatter(self.outliers[:, 0], self.outliers[:, 1], c='red', s=12, edgecolor='k')
        c = subplot.scatter(self.inliers[:, 0], self.inliers[:, 1], c='white', s=12, edgecolor='k')
        subplot.axis('tight')
        subplot.legend(
            [a.collections[0], b, c],
            ['Borda da funcao', 'Outliers', 'Inliers'],
            loc='lower right')
        subplot.set_xlim((-70, 70))
        subplot.set_ylim((-70, 70))
        plt.suptitle("Angular based outlier detection")
        plt.show()
