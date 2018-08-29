from math import ceil

import re
from keras_preprocessing.text import text_to_word_sequence
from os import path

import argparse
from keras.utils import to_categorical

import numpy as np
import logging
from scipy import sparse

from models import get_model
from preprocessing.fasttext import FastText
from preprocessing.reader import SemEvalDatasetReader, EvalitaDatasetReader, EvalitaPreprocDatasetReader
from utils import get_model_memory_usage
from utils.callbacks import EvalCallback, ValidationEarlyStopping

logging.getLogger().setLevel(logging.INFO)


class FileProvider:
    def __init__(self, workdir):
        self.model = path.join(workdir, 'model.h5')
        self.model_json = path.join(workdir, 'model.json')
        self.logs = path.join(workdir, 'logs')
        self.input_dir = path.join(workdir, 'input')
        self.semeval_train = path.join(self.input_dir, 'semeval_train')
        self.semeval_test = path.join(self.input_dir, 'semeval_test')
        self.evalita = path.join(self.input_dir, 'evalita_train.json')
        self.evalita_resolved_train = path.join(self.input_dir, 'evalita_resolved_train.json')
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
    parser.add_argument('--max-epoch', type=int, default=20,
                        help='The maximum epoch number')
    parser.add_argument('--max-seq-length', type=int, default=40,
                        help='Maximum sequence length')
    parser.add_argument('--max-char-length', type=int, default=5000,
                        help='Maximum character length')

    args = parser.parse_args()
    files = FileProvider(args.workdir)
    logging.info("Starting training with parameters: {0}".format(vars(args)))

    """##### Loading the dataset"""

    if args.semeval:
        raw_train = SemEvalDatasetReader(files.semeval_train)
        raw_test = SemEvalDatasetReader(files.semeval_test)
    else:
        raw_train, raw_test = EvalitaDatasetReader(files.evalita).split()
    raw_train, raw_val = raw_train.split(test_size=0.1)

    """##### Initializing embeddings"""
    logging.info("Initializing embeddings")

    fasttext = FastText(args.embeddings)
    embedding_size = fasttext.size
    embedding_matrix = fasttext.embeddings
    logging.info("Restored %d embeddings" % (fasttext.num_original_vectors))

    #def text_to_word_sequence(text):
    #    return re.compile("\s+").split(text.lower().strip())

    max_seq_length = 0
    max_char_length = 0
    max_perm_char_length = 0
    for text in raw_train.X:
        words = text_to_word_sequence(text)
        ngrams = list()
        for word in words:
            ngrams += fasttext.compute_ngrams(word)
        if len(words) > max_seq_length:
            max_seq_length = len(words)
        if len(ngrams) > max_char_length:
            max_char_length = len(ngrams)
        if len(ngrams) > max_perm_char_length and len(words) <= args.max_seq_length:
            max_perm_char_length = len(ngrams)
    logging.info("Max lengths in training set. seq: %d char: %d" % (max_seq_length, max_char_length))
    max_seq_length = min(max_seq_length, args.max_seq_length)
    max_char_length = min(max_char_length, args.max_char_length)

    def convert_input(raw_input):
        output = []
        output_mask = []
        counter = 0
        for text in raw_input.X:
            inp, mask = fasttext.transform_input(text_to_word_sequence(text), max_chars=max_char_length, max_words=max_seq_length)
            output.append(inp)
            output_mask.append(mask)
            counter += 1
            if counter % 10000 == 0:
                print(text, len(text))
                print(inp[:50], inp.shape)
                print("1st row", mask[0,:50], mask.shape)
                print("2nd row", mask[1,:50], mask.shape)
                print("  Converting input %d / %d" % (counter, len(raw_input.X)))
        output = np.stack(output)
        output_mask = np.stack(output_mask)
        print("Input shapes: %s %s (type: %s, size: %.1fGB)" %
              (str(output.shape), str(output_mask.shape), str(output_mask.dtype), float(output_mask.nbytes) / (1024 ** 3)))
        return output, output_mask, raw_input.Y

    X_train, X_train_mask, Y_train = convert_input(raw_train)
    X_val, X_val_mask, Y_val = convert_input(raw_val)
    X_test, X_test_mask, Y_test = convert_input(raw_test)

    Y_dictionary = raw_train.Y_dictionary
    Y_class_weights = len(Y_train) / np.power(np.bincount(Y_train), 1.1)
    Y_class_weights *= 1.0 / np.min(Y_class_weights)
    logging.info("Class weights: %s" % str(Y_class_weights))

    del raw_train
    del raw_val
    del raw_test

    # Rescaling embeddings
    means = np.mean(embedding_matrix, axis=0)
    variance = np.sqrt(np.mean((embedding_matrix - means) ** 2, axis=0))
    embedding_matrix = (embedding_matrix - means) / variance

    """##### Model definition"""
    logging.info("Initializing model")

    params = {
        "vocabulary_size": fasttext.num_original_vectors,
        "embedding_size": embedding_size,
        "max_seq_length": max_seq_length,
        "max_char_length": max_char_length,
        "embedding_matrix": embedding_matrix,
        "y_dictionary": Y_dictionary
    }
    model = get_model("base_lstm_subword").apply(params)

    """##### Load model"""

    # needs also storing&restoring of the current epoch, also not sure Adam weights are preserved
    #if path.exists(files.model):
    #    model.load_weights(files.model)

    """##### Continue with model"""

    print(model.summary())

    Y_train_one_hot = to_categorical(Y_train, num_classes=len(Y_dictionary))

    callbacks = {
        "test": EvalCallback("test", [X_test, X_test_mask], Y_test),
        #"train": EvalCallback("train", [X_train, X_train_mask], Y_train, period=3),
        "val": EvalCallback("validation", [X_val, X_val_mask], Y_val)
    }
    callbacks["stop"] = ValidationEarlyStopping(monitor=callbacks["val"])

    def generator():
        while True:
            pointer = 0
            batch_size = args.batch_size
            while pointer+batch_size < X_train.shape[0]:
                yield ({'main_input': X_train[pointer:pointer+batch_size], 'mask_input': X_train_mask[pointer:pointer+batch_size]}, Y_train_one_hot[pointer:pointer+batch_size])
                pointer += batch_size
            yield ({'main_input': X_train[pointer:], 'mask_input': X_train_mask[pointer:]}, Y_train_one_hot[pointer:])

    usage, shapes, trainable, non_trainable = get_model_memory_usage(args.batch_size, model)
    print("Model memory usage: %2fGB (%2f, %2f, %2f)" % (usage, shapes, trainable, non_trainable))

    model.fit_generator(generator(),
        steps_per_epoch=ceil(X_train.shape[0] / args.batch_size),
        class_weight=Y_class_weights,
        epochs=args.max_epoch,
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
