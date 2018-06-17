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
        elif self.dataset == 3:
            return self.tao()
        else:
            raise AttributeError("Choose a valid dataset!")

    def florest_Cover(self):
        path = "dataset/covtypeNorm.csv"
        data = pd.read_csv(path, header=None).iloc[:50000, :]
        cl2 = data.loc[data[54] == 2]
        cl4 = data.loc[data[54] == 4]
        cl2.drop(cl2.columns[54], axis=1, inplace=True)
        cl4.drop(cl4.columns[54], axis=1, inplace=True)
        inliers = cl2
        outliers = cl4
        return inliers.values, outliers.values

    def tao(self):
        number_outlier = 10
        path = "dataset/tao.csv"
        data = pd.read_csv(path, header=None).iloc[:2000, :]
        outliers = np.random.uniform(low=-60, high=60, size=(number_outlier, 4))
        return data.values, outliers

    def gauss(self):
        number_inlier = 10000
        number_outlier = 10
        offset = 10

        np.random.seed(42)
        data_1 = 10 * np.random.randn(number_inlier // 2, 2) - offset
        data_2 = 10 * np.random.randn(number_inlier // 2, 2) + offset
        inliers = np.r_[data_1, data_2]
        outliers = np.random.uniform(low=-60, high=60, size=(number_outlier, 2))
        return inliers, outliers
