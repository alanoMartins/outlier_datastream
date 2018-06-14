import numpy as np
from streamer_outlier import EllipticEnvelopeOutlierStream
from data_generator import DataGenerator

data_gen = DataGenerator(2)
inliers, outliers = data_gen.generate()
# idx = int(len(inliers) * .8)
# cl2_1 = inliers[0:idx]
# cl2_2 = inliers[idx:]


new_data = np.concatenate((inliers, outliers), axis=0)

stream = EllipticEnvelopeOutlierStream(new_data, new_data)
stream.run(new_data)
