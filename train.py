from keras.optimizers import Adam
from os import path

import argparse
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import Callback

import numpy as np
import sklearn.metrics as metrics
import logging

from models import get_model
from preprocessing.embeddings import restore_from_file
from preprocessing.reader import SemEvalDatasetReader, EvalitaDatasetReader
from preprocessing.text import Tokenizer

logging.getLogger().setLevel(logging.INFO)


class EvalCallback(Callback):
    def __init__(self, name, X_test, Y_test):
        super(EvalCallback, self).__init__()

        self.name = name
        self.X_test = X_test
        self.Y_test = Y_test

    def evaluate(self):
        Y_test_pred = [np.argmax(prediction) for prediction in self.model.predict(self.X_test)]
        logging.info("[%10s] Accuracy: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f" % (
            self.name,
            metrics.accuracy_score(self.Y_test, Y_test_pred),
            metrics.precision_score(self.Y_test, Y_test_pred, average="macro"),
            metrics.recall_score(self.Y_test, Y_test_pred, average="macro"),
            metrics.f1_score(self.Y_test, Y_test_pred, average="macro")
        ))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.evaluate()

class FileProvider:
    def __init__(self, workdir):
        self.model = path.join(workdir, 'model.h5')
        self.model_json = path.join(workdir, 'model.json')
        self.logs = path.join(workdir, 'logs')
        self.input_dir = path.join(workdir, 'input')
        self.semeval_train = path.join(self.input_dir, 'semeval_train')
        self.semeval_test = path.join(self.input_dir, 'semeval_test')
        self.evalita = path.join(self.input_dir, 'evalita_train.json')
        self.evalita_train = path.join(self.input_dir, 'evalita_split_train.json')
        self.evalita_test = path.join(self.input_dir, 'evalita_split_test.json')


if __name__ == '__main__':
    """##### Parameter parsing"""

    parser = argparse.ArgumentParser(description='Train the emoji task')
    parser.add_argument('--embeddings', default=None,
                        help='The directory with the precomputed embeddings')
    parser.add_argument('--workdir', required=True,
                        help='Work path')
    parser.add_argument('--evalita', default=False, action='store_true', help='Train and test on EVALITA2018 dataset')
    parser.add_argument('--semeval', default=False, action='store_true', help='Train and test on SemEval2018 dataset')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='The size of a mini-batch')
    parser.add_argument('--max-dict', type=int, default=300000,
                        help='Maximum dictionary size')
    parser.add_argument('--max-epoch', type=int, default=20,
                        help='The maximum epoch number')
    parser.add_argument('--max-seq-length', type=int, default=40,
                        help='Maximum sequence length')

    args = parser.parse_args()
    files = FileProvider(args.workdir)
    logging.info("Starting training with parameters: {0}".format(vars(args)))

    """##### Loading the dataset"""

    if args.semeval:
        raw_train = SemEvalDatasetReader(files.semeval_train)
        raw_test = SemEvalDatasetReader(files.semeval_test)
    else:
        if not path.exists(files.evalita_train):
            raw_train, raw_test = EvalitaDatasetReader(files.evalita).split()
        else:
            raw_train = EvalitaDatasetReader(files.evalita_train)
            raw_test = EvalitaDatasetReader(files.evalita_test)

    tokenizer = Tokenizer(num_words=args.max_dict, lower=True)
    tokenizer.fit_on_texts(raw_train.X)
    vocabulary_size = min(len(tokenizer.word_index), args.max_dict)
    logging.info("Vocabulary size: %d, Total words: %d" % (vocabulary_size, len(tokenizer.word_counts)))

    X_train = tokenizer.texts_to_sequences(raw_train.X)
    Y_train = raw_train.Y
    X_test = tokenizer.texts_to_sequences(raw_test.X)
    Y_test = raw_test.Y
    Y_dictionary = raw_train.Y_dictionary
    Y_class_weights = len(Y_train) / np.power(np.bincount(Y_train), 1.3)
    Y_class_weights *= 1.0 / np.min(Y_class_weights)
    logging.info("Class weights: %s" % str(Y_class_weights))

    del raw_train
    del raw_test

    logging.info("Padding train and test")

    max_seq_length = 0
    for seq in X_train:
        if len(seq) > max_seq_length:
            max_seq_length = len(seq)
    logging.info("Max sequence length in training set: %d" % max_seq_length)
    max_seq_length = min(max_seq_length, args.max_seq_length)
    X_train = sequence.pad_sequences(X_train, maxlen=max_seq_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_seq_length)

    """##### Initializing embeddings"""
    logging.info("Initializing embeddings")

    embedding_size = 300
    embeddings = None
    if args.embeddings:
        # Init embeddings here
        words = set(tokenizer.word_index.keys())
        embeddings, embedding_size = restore_from_file(args.embeddings, words, lower=True)

    # ReLU Xavier initialization
    embedding_matrix = np.random.randn(vocabulary_size, embedding_size).astype(np.float32) * np.sqrt(2.0/vocabulary_size)

    if embeddings is not None:
        restored = 0
        for word in embeddings:
            word_id = tokenizer.resolve_word(word)
            if word_id is not None:
                embedding_matrix[word_id] = embeddings[word]
                restored += 1
        logging.info("Restored %d (%.2f%%) embeddings" % (restored, (float(restored) / vocabulary_size) * 100))
    del embeddings

    """##### Model definition"""
    logging.info("Initializing model")

    params = {
        "vocabulary_size": vocabulary_size,
        "embedding_size": embedding_size,
        "max_seq_length": max_seq_length,
        "embedding_matrix": embedding_matrix,
        "y_dictionary": Y_dictionary
    }
    model = get_model("base_cnn").apply(params)

    """##### Load model"""

    # needs also storing&restoring of the current epoch, also not sure Adam weights are preserved
    #if path.exists(files.model):
    #    model.load_weights(files.model)

    """##### Continue with model"""

    print(model.summary())

    Y_train_one_hot = to_categorical(Y_train, num_classes=len(Y_dictionary))

    callbacks = {
        "test": EvalCallback("test", X_test, Y_test),
        "train": EvalCallback("train", X_train, Y_train)
    }
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

    callbacks["test"].evaluate()
