import argparse
import logging
import pickle
import os
import json
import numpy as np
from sklearn import metrics
from utils.fileprovider import FileProvider
from preprocessing.reader import EvalitaDatasetReader, read_emoji_dist
from keras.models import model_from_json
from keras.preprocessing import sequence

logging.getLogger().setLevel(logging.INFO)


def get_label_name(dictionary, label_number: int) -> str:
    for label_name, label_value in dictionary.items():
        if label_value == label_number:
            return label_name


def process_input(tokenizer, X, user_data=None):
    if user_data is None:
        return [tokenizer.texts_to_sequences([text for text, uid in X])]

    texts = []
    history = []

    for text, uid, tid in X:
        texts.append(text)
        if uid in user_data:
            distr = user_data[uid]
        else:
            distr = np.zeros([user_data_size], dtype=np.float16)
        history.append(distr)

    return [tokenizer.texts_to_sequences(texts), np.array(history)]


def export_predictions(file_path, predictions, raw_input):
    with open(file_path, "w") as predictions_file:
        len_labels = len(predictions[0])
        for row_index in range(0, len(predictions)):
            output_row = dict()
            output_row["tid"] = "{}".format(raw_input.X[row_index][2])  # because tuple (tweet, uid, tid)
            row_pred_asc_ord = np.argsort(predictions[row_index])  # row_predictions in asc order
            assert len_labels == len(row_pred_asc_ord)
            for label_index in reversed(range(0, len_labels)):
                output_row["label_{}".format(len_labels - label_index)] = "{}".format(
                    get_label_name(Y_dictionary, row_pred_asc_ord[label_index]))
            predictions_file.write(json.dumps(output_row))
            predictions_file.write("\n")


if __name__ == "__main__":
    """##### Parameter parsing"""

    parser = argparse.ArgumentParser(description="Predictions comparison for EVALITA2018 ITAmoji task")

    parser.add_argument('--workdir',
                        required=True,
                        help='Work path')
    parser.add_argument('--use-history',
                        required=True,
                        choices=["train", "userdata"],
                        help='Use user history to assist prediction')
    parser.add_argument("--input-dir",
                        required=True,
                        help="Input dir path")
    parser.add_argument("--folds-number",
                        type=int,
                        default=10,
                        help="Folds number")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="GPU ID to be used [0, 1, -1]")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)

    files = FileProvider(args.workdir)

    input_dir_path = args.input_dir
    folds_number = args.folds_number

    logging.info("Reading train")
    evalita_raw_train, evalita_raw_test = EvalitaDatasetReader(files.evalita).split(test_size=0.1, random_state=42)

    logging.info("Populating user history")
    # Populating user history
    user_data = None
    if args.use_history:
        if args.use_history == "userdata":
            user_data, user_data_size = read_emoji_dist(files.evalita_emoji_dist)
            user_data_size = len(user_data_size)
        else:
            user_data = {}
            user_data_size = len(evalita_raw_train.Y_dictionary)
            for i in range(len(evalita_raw_train.Y)):
                uid = evalita_raw_train.X[i][1]
                if uid not in user_data:
                    user_data[uid] = np.zeros([len(evalita_raw_train.Y_dictionary)], dtype=np.float16)
                user_data[uid][evalita_raw_train.Y[i]] += 1

    fold_predictions = []
    for fold_number in range(0, folds_number):
        logging.info("Working on fold: {}".format(fold_number))

        logging.info("Loading tokenizer")
        with open("{}/fold_{}/tokenizer.pickle".format(input_dir_path, fold_number), "rb") as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)

        logging.info("Loading model")
        json_file = open("{}/fold_{}/model.json".format(input_dir_path, fold_number), "r")
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("{}/fold_{}/model.h5".format(input_dir_path, fold_number))

        max_seq_length = model.layers[0].output_shape[1]

        logging.info("Processing train")
        X_train = process_input(tokenizer, evalita_raw_train.X, user_data)
        logging.info("Processing test")
        X_test = process_input(tokenizer, evalita_raw_test.X, user_data)

        Y_dictionary = evalita_raw_train.Y_dictionary

        logging.info("Padding train")
        X_train[0] = sequence.pad_sequences(X_train[0], maxlen=max_seq_length)
        logging.info("Padding test")
        X_test[0] = sequence.pad_sequences(X_test[0], maxlen=max_seq_length)

        logging.info("Making test predictions")
        test_predictions = model.predict(X_test)
        fold_predictions.append(test_predictions)

        accuracy_score = metrics.accuracy_score(evalita_raw_test.Y, [np.argmax(prediction) for prediction in test_predictions])
        precision_score = metrics.precision_score(evalita_raw_test.Y, [np.argmax(prediction) for prediction in test_predictions],
                                                  average="macro")
        recall_score = metrics.recall_score(evalita_raw_test.Y, [np.argmax(prediction) for prediction in test_predictions],
                                            average="macro")
        f1_score = metrics.f1_score(evalita_raw_test.Y, [np.argmax(prediction) for prediction in test_predictions], average="macro")
        logging.info("[%10s] Accuracy: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f" % (
            "fake_test fold_{}".format(fold_number),
            accuracy_score,
            precision_score,
            recall_score,
            f1_score
        ))

        logging.info("Exporting fake test predictions")
        export_predictions(file_path="{}/fold_{}/fake_test_predictions.json".format(input_dir_path, fold_number),
                           predictions=test_predictions,
                           raw_input=evalita_raw_test)

    real_test_folds_average_predictions = np.zeros(fold_predictions[0].shape)
    for fold_prediction in fold_predictions:
        real_test_folds_average_predictions = np.add(real_test_folds_average_predictions, fold_prediction)
    real_test_folds_average_predictions = real_test_folds_average_predictions / folds_number

    logging.info("Exporting real test average predictions")
    export_predictions(file_path="{}/fake_test_average_predictions.json".format(input_dir_path),
                       predictions=real_test_folds_average_predictions,
                       raw_input = evalita_raw_test)

    accuracy_score = metrics.accuracy_score(evalita_raw_test.Y, [np.argmax(prediction) for prediction in real_test_folds_average_predictions])
    precision_score = metrics.precision_score(evalita_raw_test.Y, [np.argmax(prediction) for prediction in real_test_folds_average_predictions], average="macro")
    recall_score = metrics.recall_score(evalita_raw_test.Y, [np.argmax(prediction) for prediction in real_test_folds_average_predictions], average="macro")
    f1_score = metrics.f1_score(evalita_raw_test.Y, [np.argmax(prediction) for prediction in real_test_folds_average_predictions], average="macro")
    logging.info("[%10s] Accuracy: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f" % (
        "fake_test",
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    ))





