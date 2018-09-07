import argparse
import os
import logging
import numpy as np
import subprocess
import json
import pickle
logging.getLogger().setLevel(logging.INFO)

from utils.fileprovider import FileProvider
from preprocessing.reader import EvalitaDatasetReader, read_emoji_dist
from sklearn.model_selection import StratifiedKFold
from preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from preprocessing.embeddings import restore_from_file
from models import get_model
from keras.utils import to_categorical
from keras.utils.training_utils import multi_gpu_model
from utils.callbacks import EvalCallback, ValidationEarlyStopping
from os import path


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


if __name__ == "__main__":
    """##### Parameter parsing"""

    parser = argparse.ArgumentParser(description="Cross Validation for EVALITA2018 ITAmoji task")

    parser.add_argument("--embeddings",
                        default=None,
                        help="The directory with the precomputed embeddings")
    parser.add_argument("--workdir",
                        required=True,
                        help="Work path")
    parser.add_argument("--base-model",
                        required=True,
                        choices=["base_lstm", "base_lstm_cnn"],
                        help="Model to be trained")
    parser.add_argument("--batch-size",
                        type=int,
                        default=256,
                        help="The size of a mini-batch")
    parser.add_argument("--max-epoch",
                        type=int,
                        default=40,
                        help="The maximum epoch number")
    parser.add_argument("--embeddings-only",
                        default=False,
                        action="store_true",
                        help="Only use words from the embeddings vocabulary")
    parser.add_argument("--embeddings-size",
                        type=int,
                        default=300,
                        help="Default size of the embeddings if precomputed ones are omitted")
    parser.add_argument("--embeddings-skip-first-line",
                        default=True,
                        action="store_false",
                        help="Skip first line of the embeddings")
    parser.add_argument("--max-dict",
                        type=int,
                        default=300000,
                        help="Maximum dictionary size")
    parser.add_argument("--max-seq-length",
                        type=int,
                        default=50,
                        help="Maximum sequence length")
    parser.add_argument("--use-history",
                        choices=["train", "userdata"],
                        help="Use user history to assist prediction")
    parser.add_argument("--n-folds",
                        type=int,
                        default=10,
                        help="Use user history to assist prediction")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="GPU ID to be used [0, 1, -1]")

    args = parser.parse_args()

    #os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)

    files = FileProvider(args.workdir)

    logging.info("Starting training with parameters: {0}".format(vars(args)))

    assert path.exists(files.evalita), "Unable to find {}".format(files.evalita)

    raw_train = EvalitaDatasetReader(files.evalita)
    random_state = 42
    raw_train, raw_test = raw_train.split(test_size=0.1, random_state=random_state)

    logging.info("Populating user history")
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

    logging.info("Processing input")
    raw_train.X = np.array(raw_train.X)

    skf = StratifiedKFold(n_splits=args.n_folds, random_state=random_state)
    fold_number = 0
    skf_split = list(skf.split(raw_train.X, raw_train.Y))
    while fold_number < skf.get_n_splits(raw_train.X, raw_train.Y):

        train_index, val_index = skf_split[fold_number]
        logging.info("Working on fold: {}".format(fold_number))

        fold_dir = "{}_{}/{}".format(args.base_model, args.use_history, "fold_{}".format(fold_number))

        assert (subprocess.call("mkdir -p {}/{}".format(args.workdir, fold_dir), shell=True) == 0), "unable to mkdir"

        files.model = path.join(args.workdir, fold_dir, "model.h5")
        files.model_json = path.join(args.workdir, fold_dir, "model.json")

        X_train, X_val = raw_train.X[train_index], raw_train.X[val_index]
        Y_train, Y_val = raw_train.Y[train_index], raw_train.Y[val_index]

        tokenizer = Tokenizer(num_words=args.max_dict, lower=True)
        tokenizer.fit_on_texts([text for text, uid, tid in X_train])
        vocabulary_size = min(len(tokenizer.word_index) + 1, args.max_dict)
        logging.info("Vocabulary size: %d, Total words: %d" % (vocabulary_size, len(tokenizer.word_counts)))

        logging.info("Dumping Tokenizer")
        with open("{}/{}/{}".format(args.workdir, fold_dir, "tokenizer.pickle"), "wb") as tokenizer_pickle_file:
            pickle.dump(tokenizer, tokenizer_pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        X_train = process_input(tokenizer, X_train, user_data)
        X_val = process_input(tokenizer, X_val, user_data)

        X_test = process_input(tokenizer, raw_test.X, user_data)
        Y_test = raw_test.Y

        Y_dictionary = raw_train.Y_dictionary
        Y_class_weights = len(Y_train) / np.power(np.bincount(Y_train), 1.1)
        Y_class_weights *= 1.0 / np.min(Y_class_weights)
        logging.info("Class weights: %s" % str(Y_class_weights))

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

        """##### Initializing embeddings"""
        logging.info("Initializing embeddings")

        embedding_size = args.embeddings_size
        embeddings = None
        if args.embeddings:
            # Init embeddings here
            words = set(tokenizer.word_index.keys())
            embeddings, embedding_size = restore_from_file(args.embeddings, words, lower=True,
                                                           skip_first_line=args.embeddings_skip_first_line)

        if embeddings is not None and args.embeddings_only:
            resolved = []
            for word in embeddings:
                word_id = tokenizer.resolve_word(word)
                if word_id is not None:
                    resolved.append(embeddings[word])
            logging.info("Restored %d embeddings" % len(resolved))
            embedding_matrix = np.vstack(resolved)
            vocabulary_size = len(resolved)
            del resolved
        else:
            # ReLU Xavier initialization
            embedding_matrix = np.random.randn(vocabulary_size, embedding_size).astype(np.float32)  # * np.sqrt(2.0/vocabulary_size)

            if embeddings is not None:
                restored = 0
                for word in embeddings:
                    word_id = tokenizer.resolve_word(word)
                    if word_id is not None:
                        embedding_matrix[word_id] = embeddings[word]
                        restored += 1
                logging.info("Restored %d (%.2f%%) embeddings" % (restored, (float(restored) / vocabulary_size) * 100))
        del embeddings

        # Rescaling embeddings
        means = np.mean(embedding_matrix, axis=0)
        variance = np.sqrt(np.mean((embedding_matrix - means) ** 2, axis=0))
        embedding_matrix = (embedding_matrix - means) / variance

        """##### Model definition"""
        logging.info("Initializing model")

        params = {
            "vocabulary_size": vocabulary_size,
            "embedding_size": embedding_size,
            "max_seq_length": max_seq_length,
            "embedding_matrix": embedding_matrix,
            "y_dictionary": Y_dictionary
        }

        model_name = args.base_model
        if args.use_history:
            model_name += "_user"
            params["history_size"] = user_data_size
        model, multi_model = get_model(model_name).apply(params)

        print(model.summary())

        Y_train_one_hot = to_categorical(Y_train, num_classes=len(Y_dictionary))

        callbacks = {
            "test": EvalCallback("test", X_test, Y_test),
            "train": EvalCallback("train", X_train, Y_train, period=5),
            "val": EvalCallback("validation", X_val, Y_val)
        }

        callbacks["stop"] = ValidationEarlyStopping(monitor=callbacks["val"])

        if multi_model is not None:
            print("MULTIMODEL")
            multi_model.fit(X_train,
                            Y_train_one_hot,
                            class_weight=Y_class_weights,
                            epochs=args.max_epoch,
                            batch_size=args.batch_size,
                            shuffle=True,
                            callbacks=[callback for callback in callbacks.values()])
        else:
            model.fit(X_train,
                      Y_train_one_hot,
                      class_weight=Y_class_weights,
                      epochs=args.max_epoch,
                      batch_size=args.batch_size,
                      shuffle=True,
                      callbacks=[callback for callback in callbacks.values()])

        logging.info("Saving model to json")

        model_json = model.to_json()
        with open(files.model_json, "w", encoding="utf-8") as json_file:
            json_file.write(model_json)

        logging.info("Saving model weights")

        model.save_weights(files.model)

        logging.info("Evaluating")

        f1_score = callbacks["test"].evaluate()

        delta = 0.015 * 0.44
        if f1_score < (0.44 - delta):
            print("here")
            continue
        else:
            fold_number += 1

        # exporting predictions
        logging.info("Making predictions on the test set")
        predictions = model.predict(X_test)
        assert len(raw_test.X) == len(predictions)

        logging.info("Exporting predictions on the test set")
        with open("{}/{}/predictions.json".format(args.workdir, fold_dir), "w") as predictions_file:
            len_labels = len(predictions[0])
            for row_index in range(0, len(predictions)):
                output_row = dict()
                output_row["tid"] = "{}".format(raw_test.X[row_index][2])  # because tuple (tweet, uid, tid)
                row_pred_asc_ord = np.argsort(predictions[row_index])  # row_predictions in asc order
                assert len_labels == len(row_pred_asc_ord)
                for label_index in reversed(range(0, len_labels)):
                    output_row["label_{}".format(len_labels - label_index)] = "{}".format(
                        get_label_name(Y_dictionary, row_pred_asc_ord[len_labels - label_index - 1]))
                predictions_file.write(json.dumps(output_row))
                predictions_file.write("\n")