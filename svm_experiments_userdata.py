from preprocessing.reader import EvalitaDatasetReader, read_emoji_dist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.fileprovider import FileProvider
from scipy.sparse import csr_matrix, hstack
import argparse
import string
import re
import logging
import numpy as np

logging.getLogger().setLevel(logging.INFO)


def process_text(texts, exclude=set(string.punctuation)):
    res = []
    for text in texts:
        text = re.sub(r'http\S+', '', text)
        text = text.lower()
        res.append(''.join(ch for ch in text if ch not in exclude))
    return res


def normalize(val, min_val, max_val):
    num = val - min_val
    den = max_val - min_val
    res = num /(float(den))
    return res


parser = argparse.ArgumentParser(description='Train the emoji task')
parser.add_argument('--workdir', required=False, help='Work path', default='data')
parser.add_argument('--max-dict', type=int, default=100000, help='Maximum dictionary size')
parser.add_argument('--use-history', choices=["train", "userdata"], help='Use user history to assist prediction', default='userdata')
args = parser.parse_args()
files = FileProvider(args.workdir)

logging.info("Collecting tweets")

raw_train, raw_test = EvalitaDatasetReader(files.evalita).split()

logging.info("Collecting user data")

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

logging.info("Normalizing")

minimum = user_data[raw_train.X[0][1]][0]
maximum = user_data[raw_train.X[0][1]][0]

for key, value in user_data.items():
    for elem in value:
        if elem < minimum:
            minimum = elem
        if elem > maximum:
            maximum = elem

for key, value in user_data.items():
    temp_list = []
    for elem in value:
        temp_list.append(normalize(elem, minimum, maximum))
    new_vector = np.array(temp_list)
    user_data[key] = new_vector

logging.info("Extracting data")

texts_train = []
texts_test = []
labels_train = []
labels_test = []

user_ids_train = []
user_ids_test = []

for elem in raw_train.X:
    texts_train.append(elem[0])
    user_ids_train.append(elem[1])

for elem in raw_test.X:
    texts_test.append(elem[0])
    user_ids_test.append(elem[1])

for elem in raw_train.Y:
    labels_train.append(elem)

for elem in raw_test.Y:
    labels_test.append(elem)

del raw_train

texts_train = process_text(texts_train)

vectorizer = TfidfVectorizer()

logging.info("Vectorizing")

tfidf_matrix_train = vectorizer.fit_transform(texts_train)
tfidf_matrix_test = vectorizer.fit_transform(texts_test)

del texts_train
del texts_test

complete_matrix_train = hstack([tfidf_matrix_train, csr_matrix(user_data[user_ids_train])])
complete_matrix_test = hstack([tfidf_matrix_test, csr_matrix(user_data[user_ids_test])])

logging.info("Fitting")

clf = SVC(verbose=True)
clf.fit(complete_matrix_train, labels_train)

logging.info("Predicting")

prediction = clf.predict(complete_matrix_test)

logging.info("Dumping scores")

scores_file = open('scores_file_userdata.txt', 'w')

scores_file.write('Accuracy: ' + str(accuracy_score(prediction, labels_test)) + '\n')
scores_file.write('Precision: ' + str(precision_score(prediction, labels_test, average='macro')) + '\n')
scores_file.write('Recall: ' + str(recall_score(prediction, labels_test, average='macro')) + '\n')
scores_file.write('F1-score: ' + str(f1_score(prediction, labels_test, average='macro')) + '\n')

scores_file.close()

logging.info("Done!")

