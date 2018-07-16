from os import path

import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Input, Conv1D, MaxPooling1D, Flatten, BatchNormalization, SpatialDropout1D
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras import regularizers
from keras.callbacks import Callback
from keras.preprocessing.text import Tokenizer
from keras.models import load_model, Model

import numpy as np
import json
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import logging
logging.getLogger().setLevel(logging.INFO)


class EvalCallback(Callback):
    def __init__(self, name, X_test, Y_test):
        super(EvalCallback, self).__init__()

        self.name = name
        self.X_test = X_test
        self.Y_test = Y_test

    def evaluate(self):
        Y_test_pred = [np.argmax(prediction) for prediction in self.model.predict(self.X_test)]
        logging.info("[%s] Accuracy: %.4f" % (self.name, metrics.accuracy_score(self.Y_test, Y_test_pred)))
        logging.info("[%s] Macro-Precision: %.4f" % (self.name, metrics.precision_score(self.Y_test, Y_test_pred, average="macro")))
        logging.info("[%s] Macro-Recall: %.4f" % (self.name, metrics.recall_score(self.Y_test, Y_test_pred, average="macro")))
        logging.info("[%s] Macro-F1: %.4f" % (self.name, metrics.f1_score(self.Y_test, Y_test_pred, average="macro")))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.evaluate()

class FileProvider:
    def __init__(self, workdir):
        self.model = path.join(workdir, 'model.h5')
        self.model_json = path.join(workdir, 'model.json')
        self.input_dir = path.join(workdir, 'input')
        self.semeval_train = path.join(self.input_dir, 'semeval_train')
        self.semeval_test = path.join(self.input_dir, 'semeval_test')


class SemEvalDatasetReader:
    def __init__(self, dataset_path: str):
        X_path = dataset_path+".text"
        Y_path = dataset_path+".labels"

        logging.info("Loading SemEval dataset: %s" % dataset_path)
        self.X = self._load_texts(X_path)
        self.Y, self.Y_dictionary = self._load_labels(Y_path)
        assert len(self.X) == len(self.Y)
        logging.info("Loaded %d samples for dataset %s" % (len(self.Y), dataset_path))

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


class EvalitaDatasetReader:
    def __init__(self, dataset_path: str):
        logging.info("Loading Evalita dataset: %s" % dataset_path)
        self.X, self.Y, self.Y_dictionary = self._load(dataset_path)
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


if __name__ == '__main__':
    """##### Parameter parsing"""

    parser = argparse.ArgumentParser(description='Train the emoji task')
    parser.add_argument('--embeddings', default=None,
                        help='The directory with the precomputed embeddings')
    parser.add_argument('--workdir', required=True,
                        help='Work path')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='The size of a mini-batch')
    parser.add_argument('--max-dict', type=int, default=300000,
                        help='Maximum dictionary size')
    parser.add_argument('--max-epoch', type=int, default=5,
                        help='The maximum epoch number')
    parser.add_argument('--max-seq-length', type=int, default=40,
                        help='Maximum sequence length')

    args = parser.parse_args()
    files = FileProvider(args.workdir)
    logging.info("Starting training with parameters:", vars(args))

    """##### Loading the dataset"""

    semeval_train = SemEvalDatasetReader(files.semeval_train)
    semeval_test = SemEvalDatasetReader(files.semeval_test)

    tokenizer = Tokenizer(num_words=args.max_dict, lower=True, oov_token="<unk>")
    tokenizer.fit_on_texts(semeval_train.X)
    vocabulary_size = max(len(tokenizer.word_index), args.max_dict)
    logging.info("Vocabulary size: %d, Total words: %d" % (vocabulary_size, len(tokenizer.word_counts)))

    X_train = tokenizer.texts_to_sequences(semeval_train.X)
    Y_train = semeval_train.Y
    X_test = tokenizer.texts_to_sequences(semeval_test.X)
    Y_test = semeval_test.Y
    Y_dictionary = semeval_train.Y_dictionary

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

    embedding_size = 300
    if args.embeddings:
        # Init embeddings here
        pass

    # ReLU Xavier initialization
    embedding_matrix = np.random.randn(vocabulary_size, embedding_size).astype(np.float32) * np.sqrt(2.0/(vocabulary_size))

    """##### Model definition"""

    DROPOUT_RATE = 0.4

    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_size,
                        output_dim=embedding_size,
                        weights=[embedding_matrix],
                        input_length=max_seq_length,
                        trainable=True))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Conv1D(filters=512, kernel_size=5, activation='relu', padding="same"))
    model.add(Dropout(DROPOUT_RATE))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Flatten())
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(len(Y_dictionary), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    """##### Load model"""

    if path.exists(files.model):
        model.load_weights(files.model)

    """##### Continue with model"""

    print(model.summary())

    Y_train_one_hot = to_categorical(Y_train, num_classes=len(Y_dictionary))

    callbacks = {
        "test": EvalCallback("test", X_test, Y_test),
        "train": EvalCallback("train", X_train, Y_train)
    }
    model.fit(X_train,
              Y_train_one_hot,
              epochs=args.max_epoch * 2,
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
