import json
import logging
import numpy as np
from sklearn.model_selection import train_test_split

EMOJIS = {
    'red_heart': ['❤', '♥️', '♥'],
    'face_with_tears_of_joy': '😂',
    'smiling_face_with_heart_eyes': '😍',
    'winking_face': '😉',
    'smiling_face_with_smiling_eyes': '😊',
    'beaming_face_with_smiling_eyes': '😁',
    'grinning_face': ['😀', '😃'],
    'face_blowing_a_kiss': '😘',
    'smiling_face_with_sunglasses': '😎',
    'thumbs_up': '👍',
    'rolling_on_the_floor_laughing': '🤣',
    'thinking_face': '🤔',
    'blue_heart': '💙',
    'winking_face_with_tongue': '😜',
    'face_screaming_in_fear': '😱',
    'flexed_biceps': '💪',
    'face_savoring_food': '😋',
    'grinning_face_with_sweat': '😅',
    'loudly_crying_face': '😭',
    'TOP_arrow': '🔝',
    'two_hearts': '💕',
    'sun': ['☀️', '☀️', '☀'],
    'kiss_mark': '💋',
    'sparkles': '✨',
    'rose': '🌹'
}


def read_emoji_dist(path):
    user_data = {}
    dictionary = {}
    with open(path, 'r', encoding="utf-8") as reader:
        first = True
        for line in reader:
            if first:
                first = False
                continue
            line = line.rstrip().split()

            emojis = json.loads(line[1])
            user_data[line[0]] = emojis

            for emoji in emojis:
                if emoji not in dictionary:
                    dictionary[emoji] = len(dictionary)

    for user in user_data:
        data = np.zeros([len(dictionary)])
        cur_user = user_data[user]
        for emoji in cur_user:
            data[dictionary[emoji]] = cur_user[emoji]
        user_data[user] = data

    logging.info("Loaded data on %d users" % len(user_data))
    return user_data, dictionary


class DatasetReader:
    def __init__(self, X, Y, Y_dictionary):
        self.X = X
        self.Y = Y
        self.Y_dictionary = Y_dictionary

    def split(self, test_size=0.15, random_state=42):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size = test_size, random_state = random_state, stratify = self.Y)
        return DatasetReader(X_train, Y_train, self.Y_dictionary), DatasetReader(X_test, Y_test, self.Y_dictionary)


class SemEvalDatasetReader(DatasetReader):
    def __init__(self, dataset_path: str):
        super().__init__(*self._init(dataset_path))

    def _init(self, dataset_path: str) -> (list, list, dict):
        X_path = dataset_path+".text"
        Y_path = dataset_path+".labels"

        logging.info("Loading SemEval dataset: %s" % dataset_path)
        X = self._load_texts(X_path)
        Y, Y_dictionary = self._load_labels(Y_path)
        assert len(X) == len(Y)
        logging.info("Loaded %d samples for dataset %s" % (len(Y), dataset_path))
        return X, Y, Y_dictionary

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


class EvalitaDatasetReader(DatasetReader):
    def __init__(self, dataset_path: str):
        super().__init__(*self._load(dataset_path))
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
                texts.append((sample["text_no_emoji"], sample["uid"]))
                label = sample["label"]
                if label not in dictionary:
                    dictionary[label] = len(dictionary)
                labels.append(dictionary[label])

                row += 1
                if row % 10000 == 0:
                    logging.debug("  Loaded %dk texts" % (len(texts) / 1000))

        return texts, np.array(labels), dictionary


class EvalitaPreprocDatasetReader(DatasetReader):
    def __init__(self, dataset_path: str):
        super().__init__(*self._load(dataset_path))
        logging.info("Loaded %d samples with %d classes for dataset %s" % (len(self.Y), len(self.Y_dictionary), dataset_path))

    @staticmethod
    def _load(path: str) -> (list, list, dict):
        inv_emoji = dict()
        for param in EMOJIS:
            if isinstance(EMOJIS[param], list):
                for emoji in EMOJIS[param]:
                    inv_emoji[emoji] = param
            else:
                inv_emoji[EMOJIS[param]] = param

        texts = []
        labels = []
        dictionary = {}
        counts = {}

        row = 0
        conflicts = 0
        with open(path, 'r', encoding="utf-8") as reader:
            for line in reader:
                line = line.rstrip().split('\t')
                assert len(line) == 2

                text = line[1]
                filtered_text = ''
                label = None
                conflicting = False

                for char in text:
                    if char in inv_emoji:
                        if label is not None and label != inv_emoji[char]:
                            conflicting = True
                        label = inv_emoji[char]
                        continue
                    filtered_text += char

                if label is None:
                    print("Label is none: \"%s\"" % text)

                if conflicting:
                    conflicts += 1
                    continue

                texts.append(filtered_text)
                if label not in dictionary:
                    dictionary[label] = len(dictionary)
                    counts[label] = 0
                counts[label] += 1
                labels.append(dictionary[label])

                row += 1
                if row % 10000 == 0:
                    logging.debug("  Loaded %dk texts" % (len(texts) / 1000))

        print("Class conflicts:", conflicts)
        print("Class counts:", counts)
        return texts, np.array(labels), dictionary
