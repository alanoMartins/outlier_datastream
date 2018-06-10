from streamer import OutlierStream, StreamPCA
from sklearn.covariance import EllipticEnvelope


class EllipticEnvelopeOutlierStream(OutlierStream):

    def __init__(self, data):
        OutlierStream.__init__(self, data)
        self.model = EllipticEnvelope(contamination=0.26)
        self.DEBUG = True
        self.pca_plot = StreamPCA()

    def train_model(self, data):
        if self.DEBUG:
            # Using only for a visualize
            self.pca_plot.on_start(data, EllipticEnvelope(contamination=0.26))
        self.model.fit(data)

    def update_model(self, data):
        return None

    def predict_model(self, data):
        return self.model.predict([data])

    def summary(self, predictions, data_stream):
        if self.DEBUG:
            # Using only for a visualize
            self.pca_plot.on_receive(data_stream, predictions)
        print("Results: \n")
        print(predictions)