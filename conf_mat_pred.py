import argparse
import logging
import numpy as np
import json

from utils.fileprovider import FileProvider
from preprocessing.reader import EvalitaDatasetReader, read_emoji_dist
from os import path
from preprocessing.text import Tokenizer
from keras.models import model_from_json
from keras.preprocessing import sequence
from utils.plotter import Plotter

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot confusion matrix and export predictions')
    parser.add_argument('--workdir', required=True,
                        help='Work path')
    parser.add_argument('--use-history', choices=["train", "userdata"], help='Use user history to assist prediction')
    parser.add_argument('--max-dict', type=int, default=300000,
                        help='Maximum dictionary size')
    parser.add_argument('--max-seq-length', type=int, default=50,
                        help='Maximum sequence length')
    parser.add_argument('--plot-cm', default=False, action="store_true",
                        help='Plot confusion matrix')

    args = parser.parse_args()
    files = FileProvider(args.workdir)

    if not path.exists(files.evalita_train):
        raw_train, raw_test = EvalitaDatasetReader(files.evalita).split()
    else:
        raw_train = EvalitaDatasetReader(files.evalita_train)
        raw_test = EvalitaDatasetReader(files.evalita_test)
    raw_train, raw_val = raw_train.split(test_size=0.1)

    tokenizer = Tokenizer(num_words=args.max_dict, lower=True)
    tokenizer.fit_on_texts([text for text, uid, tid in raw_train.X])
    vocabulary_size = min(len(tokenizer.word_index)+1, args.max_dict)
    logging.info("Vocabulary size: %d, Total words: %d" % (vocabulary_size, len(tokenizer.word_counts)))

    # Populating user history
    user_data = None
    if args.use_history:
        if args.use_history == "userdata":
            user_data, user_data_size = read_emoji_dist(files.evalita_emoji_dist)
            user_data_size = len(user_data_size)
        else:
            user_data = {}
            user_data_size = len(raw_train.Y_dictionary)
            for i in range(len(raw_train.Y)):
                uid = raw_train.X[i][1]
                if uid not in user_data:
                    user_data[uid] = np.zeros([len(raw_train.Y_dictionary)], dtype=np.float16)
                user_data[uid][raw_train.Y[i]] += 1

    def process_input(raw_input, user_data=None):
        if user_data is None:
            return [tokenizer.texts_to_sequences([text for text, uid, tid in raw_train.X])], raw_input.Y

        texts = []
        history = []

        for text, uid, tid in raw_input.X:
            texts.append(text)
            if uid in user_data:
                distr = user_data[uid]
            else:
                distr = np.zeros([user_data_size], dtype=np.float16)
            history.append(distr)

        return [tokenizer.texts_to_sequences(texts), np.array(history)], raw_input.Y

    X_train, Y_train = process_input(raw_train, user_data)
    X_val, Y_val = process_input(raw_val, user_data)
    X_test, Y_test = process_input(raw_test, user_data)
    Y_dictionary = raw_train.Y_dictionary
    Y_class_weights = len(Y_train) / np.power(np.bincount(Y_train), 1.1)
    Y_class_weights *= 1.0 / np.min(Y_class_weights)
    logging.info("Class weights: %s" % str(Y_class_weights))

    del raw_train
    del raw_val

    logging.info("Padding train and test")

    max_seq_length = 0
    for seq in X_train[0]:
        if len(seq) > max_seq_length:
            max_seq_length = len(seq)
    logging.info("Max sequence length in training set: %d" % max_seq_length)
    max_seq_length = min(max_seq_length, args.max_seq_length)
    X_train[0] = sequence.pad_sequences(X_train[0], maxlen=max_seq_length)
    X_val[0] = sequence.pad_sequences(X_val[0], maxlen=max_seq_length)
    X_test[0] = sequence.pad_sequences(X_test[0], maxlen=max_seq_length)

    assert path.exists(files.model) and path.exists(files.model_json), "Unable to find {} and {}".format(files.model, files.model_json)

    logging.info("Loading model from disk")
    json_file = open(files.model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(files.model)

    def get_label_name(dictionary, label_number: int) -> str:
        for label_name, label_value in dictionary.items():
            if label_value == label_number:
                return label_name

    # exporting predictions
    logging.info("Making predictions on the test set")
    predictions = model.predict(X_test)
    assert len(raw_test.X) == len(predictions)

    logging.info("Exporting predictions on the test set")
    with open("{}/predictions.json".format(args.workdir), "w") as predictions_file:
        len_labels = len(predictions[0])
        for row_index in range(0, len(predictions)):
            output_row = dict()
            output_row["tid"] = "{}".format(raw_test.X[row_index][2])  # because tuple (tweet, uid, tid)
            row_pred_asc_ord = np.argsort(predictions[row_index])  # row_predictions in asc order
            assert len_labels == len(row_pred_asc_ord)
            for label_index in reversed(range(0, len_labels)):
                output_row["label_{}".format(len_labels - label_index)] = "{}".format(
                    get_label_name(Y_dictionary, row_pred_asc_ord[label_index]))
            predictions_file.write(json.dumps(output_row))
            predictions_file.write("\n")

    if args.plot_cm:
        plotter = Plotter(model, X_test[0], Y_test, args.workdir)
        logging.info("Computing and plotting confusion matrix")
        plotter.compute_and_save_confusion_matrix()