import os
import argparse
from keras.utils import to_categorical

import numpy as np
import logging

from models import get_model
from preprocessing.reader import SemEvalDatasetReader, EvalitaDatasetReader
from utils.callbacks import EvalCallback, ValidationEarlyStopping
from utils.fileprovider import FileProvider
from utils.converter import Converter

logging.getLogger().setLevel(logging.INFO)

"""
    This is a porting of the https://github.com/zonetrooper32/VDCNN
"""

if __name__ == "__main__":
    """##### Parameter parsing"""

    parser = argparse.ArgumentParser(description="Train the emoji task")

    # General parameters
    parser.add_argument("--workdir", required=True,
                        help="Work path")
    parser.add_argument("--evalita", default=False, action="store_true",
                        help="Train and test on EVALITA2018 dataset")
    parser.add_argument("--semeval", default=False, action="store_true",
                        help="Train and test on SemEval2018 dataset")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to be used [0, 1, -1]")

    # Model parameters
    parser.add_argument("--max-seq-length", type=int, default=1024,
                        help="Maximum sequence length (default: 1024)")
    parser.add_argument("--pool-type", choices=["max", "k_max", "conv"], default="max",
                        help="Types of downsampling methods [max, k_max, conv] (default: \"max\"")
    parser.add_argument("--depth", type=int, choices=[9, 17, 29, 47], default=9,
                        help="Depth for VDCNN [9, 17, 29, 49] (defalut: 9)")
    parser.add_argument("--shortcut", default=False, action="store_true",
                        help="Use optional shortcut (default: False)")
    parser.add_argument("--sorted", default=False, action="store_true",
                        help="Sort during k-max pooling (default: False)")
    parser.add_argument("--bias", default=False, action="store_true",
                        help="Use bias for all Conv1D layers (default: False)")

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=256,
                        help="The size of a mini-batch (default: 256)")
    parser.add_argument("--max-epoch", type=int, default=20,
                        help="The maximum epoch number (default: 100)")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)

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
    raw_train, raw_val = raw_train.split(test_size=0.1)

    converter = Converter(sequence_max_length=args.max_seq_length)

    X_train = converter.texts_to_sequences(raw_train.X)
    Y_train = raw_train.Y
    X_val = converter.texts_to_sequences(raw_val.X)
    Y_val = raw_val.Y
    X_test = converter.texts_to_sequences(raw_test.X)
    Y_test = raw_test.Y
    Y_dictionary = raw_train.Y_dictionary
    Y_class_weights = len(Y_train) / np.power(np.bincount(Y_train), 1.1)
    Y_class_weights *= 1.0 / np.min(Y_class_weights)
    logging.info("Class weights: %s" % str(Y_class_weights))

    del raw_train
    del raw_val
    del raw_test

    """##### Model definition"""
    logging.info("Initializing model")

    params = {
        "num_classes": len(Y_dictionary),
        "depth": args.depth,
        "sequence_length": args.max_seq_length,
        "shortcut": args.shortcut,
        "pool_type": args.pool_type,
        "sorted": args.sorted,
        "use_bias": args.bias
    }

    model = get_model("vdcnn").apply(params)

    """##### Load model"""

    # needs also storing&restoring of the current epoch, also not sure Adam weights are preserved
    #if path.exists(files.model):
    #    model.load_weights(files.model)

    """##### Continue with model"""

    print(model.summary())

    Y_train_one_hot = to_categorical(Y_train, num_classes=len(Y_dictionary))

    callbacks = {
        "test": EvalCallback("test", X_test, Y_test),
        "train": EvalCallback("train", X_train, Y_train, period=3),
        "val": EvalCallback("validation", X_val, Y_val)
    }
    callbacks["stop"] = ValidationEarlyStopping(monitor=callbacks["val"], patience=2)
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
