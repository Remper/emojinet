import argparse
import logging
import pickle
import numpy as np
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
    parser.add_argument("--max-seq-length",
                        type=int,
                        default=48,
                        help="Maximum sequence length")

    args = parser.parse_args()
    files = FileProvider(args.workdir)

    input_dir_path = args.input_dir

    logging.info("Reading train")
    evalita_raw_train = EvalitaDatasetReader(files.evalita)
    logging.info("Reading test")
    evalita_raw_real_test =  EvalitaDatasetReader(files.evalita_real_test)

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

    for fold_number in range(0, 10):
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

        logging.info("Processing train")
        X_train = process_input(tokenizer, evalita_raw_train.X, user_data)
        logging.info("Processing test")
        X_test = process_input(tokenizer, evalita_raw_real_test.X, user_data)

        Y_dictionary = evalita_raw_train.Y_dictionary

        logging.info("Padding train")
        X_train[0] = sequence.pad_sequences(X_train[0], maxlen=args.max_seq_length)
        logging.info("Padding test")
        X_test[0] = sequence.pad_sequences(X_test[0], maxlen=args.max_seq_length)

        logging.info("Making test predictions")
        train_predictions = model.predict(X_train)
        print(train_predictions)

        break



