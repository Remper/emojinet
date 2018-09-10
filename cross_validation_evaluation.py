import os
from os import path

import argparse
import json
import logging
from sklearn import metrics

logging.getLogger().setLevel(logging.INFO)


if __name__ == "__main__":
    """##### Parameter parsing"""

    parser = argparse.ArgumentParser(description="Cross Validation evaluation producer for EVALITA2018 ITAmoji task")

    parser.add_argument("--input-file",
                        default=None,
                        required=True,
                        help="Input file path")
    parser.add_argument("--train-file",
                        default=None,
                        required=True,
                        help="EVALITA train file")

    args = parser.parse_args()

    input_file_path = args.input_file
    train_file_path = args.train_file

    label_dictionary = {}
    samples = {}
    duplicates = 0
    with open(train_file_path, "r", encoding="utf-8") as train_file:
        for line in train_file:
            sample = json.loads(line.rstrip())
            if sample["tid"] in samples:
                duplicates += 1
            samples[sample["tid"]] = sample["label"]
            if sample["label"] not in label_dictionary:
                label_dictionary[sample["label"]] = len(label_dictionary)
    logging.info("Loaded %d gold standard tweets (%d duplicates)" % (len(samples), duplicates))

    folds = sorted(os.listdir(input_file_path))
    for fold in folds:
        if not fold.startswith("fold"):
            continue
        fold_dir = path.join(input_file_path, fold)

        # Loading test set predictions
        predictions = []
        gold = []
        with open(path.join(fold_dir, "fake_average_predictions.json"), "r", encoding="utf-8") as reader:
            for line in reader:
                sample = json.loads(line.rstrip())
                if sample["tid"] not in samples:
                    logging.warning("[%s] Tweet %s was not found in gold standard" % (fold, sample["tid"]))
                    continue
                predictions.append(sample["label_1"])
                gold.append(samples[sample["tid"]])
        logging.info("[%s] Loaded %d predictions" % (fold, len(predictions)))

        accuracy_score = metrics.accuracy_score(gold, predictions)
        precision_score = metrics.precision_score(gold, predictions, average="macro")
        recall_score = metrics.recall_score(gold, predictions, average="macro")
        f1_score = metrics.f1_score(gold, predictions, average="macro")
        logging.info("[%s] Accuracy: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f" % (
            fold,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score
        ))

        for file in ["real_predictions.json", "real_average_predictions.json"]:
            dist = {}
            lines = 0
            with open(path.join(fold_dir, file), "r", encoding="utf-8") as reader:
                for line in reader:
                    sample = json.loads(line.rstrip())
                    if sample["label_1"] not in dist:
                        dist[sample["label_1"]] = 0
                    dist[sample["label_1"]] += 1
                    lines += 1
            print("["+file+"] Distribution: "+"\t".join([tuple[0]+":"+str(float(tuple[1])/lines) for tuple in sorted([(label, dist[label]) for label in dist], key= lambda x : -x[1])[:3]]))