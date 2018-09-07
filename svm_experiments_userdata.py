from preprocessing.reader import EvalitaDatasetReader, read_emoji_dist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.fileprovider import FileProvider
from scipy.sparse import csr_matrix
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


def csr_vappend(a,b):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of the first one.
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""

    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])
    return a


parser = argparse.ArgumentParser(description='Train the emoji task')
parser.add_argument('--workdir', required=False, help='Work path', default='data')
parser.add_argument('--max-dict', type=int, default=100000, help='Maximum dictionary size')
parser.add_argument('--use-history', choices=["train", "userdata"], help='Use user history to assist prediction', default='userdata')
args = parser.parse_args()
files = FileProvider(args.workdir)
raw_train, raw_test = EvalitaDatasetReader(files.evalita).split()

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

texts_train = []
texts_test = []
labels_train = []
labels_test = []

user_ids_train = []

for elem in raw_train.X:
    texts_train.append(elem[0])
    user_ids_train.append(elem[1])

for elem in raw_test.X:
    texts_test.append(elem[0])

for elem in raw_train.Y:
    labels_train.append(elem)

for elem in raw_test.Y:
    labels_test.append(elem)

del raw_train

texts_train = process_text(texts_train)

vectorizer = TfidfVectorizer()

logging.info("Starting vectorization")

tfidf_matrix_train = vectorizer.fit_transform(texts_train)
tfidf_matrix_test = vectorizer.fit_transform(texts_test)

del texts_train
del texts_test

for i in range(tfidf_matrix_train.shape[0]):
    #print(csr_vappend(tfidf_matrix_train[i], csr_matrix(user_data[user_ids_train[i]])))
    print(tfidf_matrix_train[i])
    print(csr_matrix(user_data[user_ids_train[i]]))
    break

print(type(tfidf_matrix_train))

'''
logging.info("Ended vectorization. Starting svm fit")

clf = SVC()
clf.fit(tfidf_matrix_train, labels_train)

logging.info("Ended svm fit. Starting prediction")

prediction = clf.predict(tfidf_matrix_test)

logging.info("Ended prediction. Starting dump of scores")

scores_file = open('scores_file.txt', 'w')

scores_file.write('Accuracy: ' + str(accuracy_score(prediction, labels_test)) + '\n')
scores_file.write('Precision: ' + str(precision_score(prediction, labels_test)) + '\n')
scores_file.write('Recall: ' + str(recall_score(prediction, labels_test)) + '\n')
scores_file.write('F1-score: ' + str(f1_score(prediction, labels_test)) + '\n')

scores_file.close()

logging.info("Done")
'''
