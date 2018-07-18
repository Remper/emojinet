import json
import logging
import numpy as np
from sklearn.model_selection import train_test_split


class DatasetReader:
    def __init__(self, X, Y, Y_dictionary):
        self.X = X
        self.Y = Y
        self.Y_dictionary = Y_dictionary

    def split(self, test_size=0.15, random_state=42):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size = test_size, random_state = random_state, stratify = self.Y)
        return DatasetReader(X_train, Y_train, self.Y_dictionary), DatasetReader(X_test, Y_test, self.Y_dictionary)


class SemEvalDatasetReader(DatasetReader):
    def __init__(self, dataset_path: str):
        super().__init__(*self._init(dataset_path))

    def _init(self, dataset_path: str) -> (list, list, dict):
        X_path = dataset_path+".text"
        Y_path = dataset_path+".labels"

        logging.info("Loading SemEval dataset: %s" % dataset_path)
        X = self._load_texts(X_path)
        Y, Y_dictionary = self._load_labels(Y_path)
        assert len(X) == len(Y)
        logging.info("Loaded %d samples for dataset %s" % (len(Y), dataset_path))
        return X, Y, Y_dictionary

    @staticmethod
    def _load_labels(path: str) -> (list, dict):
        labels = []
        dictionary = {}
        with open(path, 'r', encoding="utf-8") as reader:
            for line in reader:
                label = int(line.rstrip())
                labels.append(label)
                dictionary[label] = label
        return np.array(labels), dictionary

    @staticmethod
    def _load_texts(path: str) -> list:
        texts = []
        row = 0
        with open(path, 'r', encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                texts.append(line)

                row += 1
                if row % 10000 == 0:
                    logging.debug("  Loaded %dk texts" % (len(texts) / 1000))
        return texts


class EvalitaDatasetReader(DatasetReader):
    def __init__(self, dataset_path: str):
        super().__init__(*self._load(dataset_path))
        logging.info("Loaded %d samples with %d classes for dataset %s" % (len(self.Y), len(self.Y_dictionary), dataset_path))

    @staticmethod
    def _load(path: str) -> (list, list, dict):
        texts = []
        labels = []
        dictionary = {}

        row = 0
        with open(path, 'r', encoding="utf-8") as reader:
            for line in reader:
                line = line.rstrip()
                sample = json.loads(line)
                texts.append(sample["text_no_emoji"])
                label = sample["label"]
                if label not in dictionary:
                    dictionary[label] = len(dictionary)
                labels.append(dictionary[label])

                row += 1
                if row % 10000 == 0:
                    logging.debug("  Loaded %dk texts" % (len(texts) / 1000))

        return texts, np.array(labels), dictionary