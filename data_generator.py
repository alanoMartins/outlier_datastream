import pandas as pd
import numpy as np


class DataGenerator:
    def __init__(self, dataset=1):
        self.dataset = dataset

    def generate(self):
        if self.dataset == 1:
            return self.florest_Cover()
        elif self.dataset == 2:
            return self.gauss()
        else:
            raise AttributeError("Choose a valid dataset!")

    def florest_Cover(self):
        path = "dataset/covtypeNorm.csv"
        data = pd.read_csv(path, header=None).iloc[:, :]
        cl2 = data.loc[data[54] == 2]
        cl4 = data.loc[data[54] == 4]
        cl2.drop(cl2.columns[54], axis=1, inplace=True)
        cl4.drop(cl4.columns[54], axis=1, inplace=True)
        inliers = cl2
        outliers = cl4
        return inliers, outliers

    def tao(self):
        pass

    def gauss(self):
        number_inlier = 1000
        number_outlier = 10
        offset = 10

        np.random.seed(42)
        data_1 = 0.3 * np.random.randn(number_inlier // 2, 2) - offset
        data_2 = 0.3 * np.random.randn(number_inlier // 2, 2) + offset
        inliers = np.r_[data_1, data_2]
        outliers = np.random.uniform(low=-6, high=6, size=(number_outlier, 2))
        return inliers, outliers