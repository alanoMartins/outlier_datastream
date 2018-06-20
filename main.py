from data_generator import DataGenerator
from outlier_detector import AngularBasedOutlier
import numpy as np
from streamer_outlier import TemplateOutlier
from pyod.models.abod import ABOD
from pyod.models.lof import LOF
from pyod.models.hbos import HBOS

data_gen = DataGenerator(1)
inliers, outliers = data_gen.generate()

experiemnts_window = [(.4, .3), (.2, .15), (.1, .1), (.1, .05)]
#experiemnts_nei = [(10, 0.0005), (20, 0.0005), (30, 0.0005), (50, 0.0005), (90, 0.0005)]
experiemnts_nei = [(300, 0.3), (300, 0.2), (400, 0.1)]
#experiemnts_nei = [(20, .1), (20, .2), (20, .3), (20, .4), (20, .5)]

results = []

# for w, s in experiemnts_window:
#     stream = AngularBasedOutlier(inliers, outliers)
#     stream.run(outliers, window_size=w, slide=s)
#     results.append(stream.stats)

for n, c in experiemnts_nei:
    model = HBOS(contamination=c)
    #model = ABOD(n_neighbors=n, contamination=c)
    stream = TemplateOutlier(inliers, outliers, model)
    stream.run(np.r_[outliers, inliers], window_size=.8, slide=.8)
    results.append(stream.stats)

times = list(map(lambda x: x["time"], results))
accuracies = list(map(lambda x: x["accuracy"], results))
precisions = list(map(lambda x: x["precision"], results))
recalls = list(map(lambda x: x["recall"], results))
f1s = list(map(lambda x: x["f1"], results))

y_axes = np.linspace(0, len(results), len(results))

print(results)

import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(211)
ax.set_title('Time Exec')


ax.set_ylabel('seg')

ax.plot(times)

ax = fig.add_subplot(212)
ax.set_title('Metrics')


ax.plot(accuracies, label='Accuracy')
ax.plot(precisions, label='Precision')
ax.plot(recalls, label='Recall')
ax.plot(f1s, label='F1')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
