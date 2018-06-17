import numpy as np
from streamer_outlier import EllipticEnvelopeOutlierStream
from data_generator import DataGenerator
from outlier_detector import AngularBasedOutlier

data_gen = DataGenerator(2)
inliers, outliers = data_gen.generate()

stream = AngularBasedOutlier(inliers, outliers)
stream.run(outliers, window_size=.4, slide=.3)
