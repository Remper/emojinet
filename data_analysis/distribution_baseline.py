import json
import logging
import argparse
import numpy as np
from random import random
from sklearn import metrics

from preprocessing.reader import DatasetReader
from utils.fileprovider import FileProvider
logging.getLogger().setLevel(logging.INFO)


if __name__ == '__main__':
    """##### Parameter parsing"""

    parser = argparse.ArgumentParser(description='A baseline based on returning the most common emoji given the user')
    parser.add_argument('--workdir', required=True, help='Work path')

    args = parser.parse_args()
    files = FileProvider(args.workdir)

    Y = []
    X = []
    users = {}
    dictionary = {}

    with open(files.evalita, 'r', encoding="utf-8") as reader:
        for line in reader:
            line = line.rstrip()
            sample = json.loads(line)
            uid = sample["uid"]
            label = sample["label"]

            if uid not in users:
                users[uid] = {}
            if label not in dictionary:
                dictionary[label] = len(dictionary)
            label = dictionary[label]

            X.append(uid)
            Y.append(label)
            if label not in users[uid]:
                users[uid][label] = 0
            users[uid][label] += 1

    for user in users:
        raw_distribution = users[user]
        distribution = np.zeros([len(dictionary)])
        for label in raw_distribution:
            distribution[label] = raw_distribution[label]
        users[user] = distribution


    raw_train, raw_test = DatasetReader(X, Y, dictionary).split(random_state=None)
    raw_test.Y = np.array(raw_test.Y)

    # Removing test set data from the distribution
    for i in range(len(raw_test.X)):
        users[raw_test.X[i]][raw_test.Y[i]] -= 1

    predictions = []
    for value in raw_test.X:
        if max(users[value]) == 0:
            predictions.append(0)
        else:
            predictions.append(np.argmax(users[value]))

    logging.info("Accuracy: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f" % (
        metrics.accuracy_score(raw_test.Y, predictions),
        metrics.precision_score(raw_test.Y, predictions, average="macro"),
        metrics.recall_score(raw_test.Y, predictions, average="macro"),
        metrics.f1_score(raw_test.Y, predictions, average="macro")
    ))