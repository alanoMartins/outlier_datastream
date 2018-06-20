import time
import numpy as np


class Streamer:

    def on_start(self, data):
        raise NotImplementedError("Please Implement this method")

    def on_receive(self, data):
        raise NotImplementedError("Please Implement this method")

    def on_finish(self):
        raise NotImplementedError("Please Implement this method")

    def __init__(self, data):
        self.data = data
        self.stats = {}

    def run(self, data_stream, window_size=.1, slide=.05):
        start = time.time()
        self.on_start(self.data)
        self.data_stream = data_stream
        absolute_size = int(len(data_stream) * window_size)
        absolute_slide = int(absolute_size * slide)
        initial_index = 0
        final_index = absolute_size

        if absolute_size == 0:
            raise AttributeError("Window size is too small")

        if absolute_slide == 0:
            raise AttributeError("Slide is too small")

        while final_index <= len(data_stream):
            if len(data_stream) <= final_index + absolute_slide:
                final_index = len(data_stream)
            window = data_stream[initial_index:final_index]
            self.on_receive(window, initial_index, final_index)
            initial_index += absolute_slide
            final_index += absolute_slide
        end = time.time()
        self.on_finish()
        self.stats["time"] = end - start


class OutlierStream(Streamer):

    def __init__(self, inliers, outliers):
        self.ground_truth = []
        [self.ground_truth.append(1) for outlier in outliers]
        [self.ground_truth.append(0) for inline in inliers]
        data_total = np.concatenate((outliers, inliers), axis=0)
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


