from streamer_outlier import TemplateOutlier
from pyod.models.lof import LOF
from pyod.models.abod import ABOD


class AngularBasedOutlier(TemplateOutlier):

    def __init__(self, inliers, outliers):
        self.model = ABOD(n_neighbors=20, contamination=0.2)
        TemplateOutlier.__init__(self, inliers, outliers, self.model)

class LOFOutlier(TemplateOutlier):

    def __init__(self, inliers, outliers):
        self.model = LOF(n_neighbors=20, contamination=0.2)
        TemplateOutlier.__init__(self, inliers, outliers, self.model)
