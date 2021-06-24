"""
    healer class that is used to read csv file and normalize data using
    quantization method
    @author      Meet Shah
"""
import math
import pandas as pd

class file_operation:

    __slots__ = "classification_training_data"

    def __init__(self):
        self.classification_training_data = None

    def read_csv(self, file_name, bin_size=1):
        """
        this function read file from csv file and store them into panda function
        :param file_name: name of csv file to read from
        :param bin_size: size of bin used for noise reduction
        :return: dataframe that stores data from
        """
        # read training data from csv file and store into dataframe in panda
        self.classification_training_data = pd.read_csv(file_name, dtype={
            "Age": float,
            "Ht": float,
            "TailLn": float,
            "HairLn": float,
            "BangLn": float,
            "Reach": float,
            "EarLobes": float,
            "Class": str
        })
        self.normalize_data()
        return self.classification_training_data

    def normalize_data(self):
        """
        Normalize data for noise reduction purpose
        :return:
        """
        # quantify data for each column except classification column for noise reduction
        for column_header in self.classification_training_data.columns:
            if column_header == "Class":
                continue
            if column_header == "Age":
                bin_size = 2
            elif column_header == "Ht":
                bin_size = 5
            else:
                bin_size = 1
            for idx in self.classification_training_data.index:
                self.classification_training_data.at[idx, column_header] = math.floor(
                    self.classification_training_data[column_header][idx] / bin_size) * bin_size

