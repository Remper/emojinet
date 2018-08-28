import argparse
import logging
import numpy as np
from scipy.spatial import distance

from preprocessing.fasttext import FastText

if __name__ == '__main__':
    """##### Parameter parsing"""

    parser = argparse.ArgumentParser(description='Train the emoji task')
    parser.add_argument('--embeddings', default=None,
                        help='The directory with the precomputed embeddings')
    args = parser.parse_args()
    logging.info("Starting training with parameters: {0}".format(vars(args)))

    fasttext = FastText(args.embeddings)
    logging.info("Restored %d embeddings" % (fasttext.bucket))

    base_word = "gatto"
    not_a_similar_word = "эмоджи"
    targets = {
        "gattino": 0.823819,
        "cane": 0.820798,
        "miciogatto": 0.774525,
        "gattone": 0.773202,
        "miciogattone": 0.761394,
        "miogatto": 0.746384,
        "gattonzo": 0.745221,
        "miagolante": 0.743607,
        "micio": 0.740949,
        "gattoiattolo": 0.728013,
    }

    def produce_word_vector(word):
        indices = fasttext.compute_ngram_indices(word)
        vectors = fasttext.embeddings[indices,:]
        return np.mean(vectors, axis=0)

    base_vector = produce_word_vector(base_word)
    for target in targets:
        vector = produce_word_vector(target)
        print("%15s\t%15s\t%.3f\t%.3f" % (base_word, target, 1-distance.cosine(base_vector, vector), targets[target]))
    not_a_similar_vector = produce_word_vector(not_a_similar_word)
    print("%15s\t%15s\t%.3f" % (base_word, not_a_similar_word, 1-distance.cosine(base_vector, not_a_similar_vector)))
