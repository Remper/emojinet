import numpy as np
import time
import gzip


def open_custom(file: str):
    if file.endswith('.gz'):
        return gzip.open(file, 'rt')
    return open(file, 'r')


def restore_from_file(file: str, words: set, lower=False) -> (dict, int):
    dictionary = dict()
    count = 0
    row_size = 0
    timestamp = time.time()
    symbol = None
    with open_custom(file) as reader:
        for line in reader:
            if symbol is None:
                for test_symbol in ['\t', ' ']:
                    row = line.rstrip().split(test_symbol)
                    if len(row) > 10:
                        symbol = test_symbol
                        row_size = len(row)
                        print("Break symbol for this embedding: '%s', embedding size: %d" % (test_symbol, row_size-1))
                        break
            if symbol is None:
                print("Can't determine break symbol, skipping")
                return None
            row = line.rstrip().split(symbol)

            if len(row) != row_size:
                print("Inconsistent row size: %d instead of %d" % (len(row), row_size))
                continue

            count += 1
            if count % 100000 == 0:
                print("  %.1fm embeddings parsed (%.3fs)" % (float(count) / 1000000, time.time() - timestamp))
                timestamp = time.time()

            word = row[0]
            if lower:
                word = word.lower()
            if word not in words:
                continue
            dictionary[word] = np.array([float(ele) for ele in row[1:]])

    print("  %.1fm embeddings parsed (%.3fs)" % (float(count) / 1000000, time.time() - timestamp))
    return dictionary, row_size-1
