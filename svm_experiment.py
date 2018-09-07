from preprocessing.reader import EvalitaDatasetReader
from sklearn.feature_extraction.text import TfidfVectorizer
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

texts = []

for elem in raw_train.X:
    texts.append(elem[0])

texts = process_text(texts)

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(texts)

dense_matrix = tfidf_matrix.toarray()


