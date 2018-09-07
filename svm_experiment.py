from preprocessing.reader import EvalitaDatasetReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.fileprovider import FileProvider
import argparse
import string
import re


def process_text(texts, exclude=set(string.punctuation)):
    res = []
    for text in texts:
        text = re.sub(r'http\S+', '', text)
        res.append(''.join(ch for ch in text if ch not in exclude))
    return res


parser = argparse.ArgumentParser(description='Train the emoji task')
parser.add_argument('--workdir', required=False, help='Work path', default='data')
parser.add_argument('--max-dict', type=int, default=100000, help='Maximum dictionary size')
args = parser.parse_args()
files = FileProvider(args.workdir)
raw_train, raw_test = EvalitaDatasetReader(files.evalita).split()

texts_train = []
texts_test = []
labels_train = []
labels_test = []

for elem in raw_train.X:
    texts_train.append(elem[0])

for elem in raw_test.X:
    texts_test.append(elem[0])
    
for elem in raw_train.Y:
    labels_train.append(elem)

for elem in raw_test.Y:
    labels_test.append()

del raw_train

texts_train = process_text(texts_train)

vectorizer = TfidfVectorizer()

tfidf_matrix_train = vectorizer.fit_transform(texts_train)
tfidf_matrix_test = vectorizer.fit_transform(texts_test)

del texts_train
del texts_test

clf = SVC()
clf.fit(tfidf_matrix_train, labels_train)

prediction = clf.predict(tfidf_matrix_test)

scores_file = open('scores_file.txt', 'w')

scores_file.write('Accuracy: ' + str(accuracy_score(prediction, labels_test)) + '\n')
scores_file.write('Precision: ' + str(precision_score(prediction, labels_test)) + '\n')
scores_file.write('Recall: ' + str(recall_score(prediction, labels_test)) + '\n')
scores_file.write('F1-score: ' + str(f1_score(prediction, labels_test)) + '\n')

scores_file.close()


