import numpy as np
import struct
from keras_preprocessing.text import text_to_word_sequence

FASTTEXT_VERSION = 12
FASTTEXT_MAGIC = 793712314

class FastText:
    """
    Utility class that is able to parse and provide character ngrams produced by the reference FastText(1) implementation
    [(1) FastText](https://github.com/facebookresearch/fastText)
    """
    def __init__(self, path):
        self.load_binary_data(path+'.bin')

    def load_binary_data(self, model_binary_file):
        """Loads data from the output binary file created by FastText training"""
        with open(model_binary_file, 'rb') as f:
            (magic, version) = self.struct_unpack(f, '@2i')
            assert magic == FASTTEXT_MAGIC
            assert version == FASTTEXT_VERSION
            self.load_model_params(f)
            self.load_dict(f)
            self.load_vectors(f)

    def load_model_params(self, file_handle):
        (dim, ws, epoch, minCount, neg, _, loss, model, bucket, minn, maxn, _, t) = self.struct_unpack(file_handle,
                                                                                                       '@12i1d')
        # Parameters stored by [Args::save](https://github.com/facebookresearch/fastText/blob/master/src/args.cc)
        self.size = dim
        self.window = ws
        self.iter = epoch
        self.min_count = minCount
        self.negative = neg
        self.hs = loss == 1
        self.sg = model == 2
        self.bucket = bucket
        self.min_n = minn
        self.max_n = maxn
        self.sample = t
        self.vocab = dict()

    def load_dict(self, file_handle):
        (vocab_size, nwords, nlabels) = self.struct_unpack(file_handle, '@3i')
        # Vocab stored by [Dictionary::save](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc)
        assert vocab_size == nwords, 'mismatch between vocab sizes'
        (ntokens, prune) = self.struct_unpack(file_handle, '@2q')
        for i in range(nwords):
            word = b''
            char = file_handle.read(1)
            # Read vocab word
            while char != b'\x00':
                word += char
                char = file_handle.read(1)
            word = word.decode('utf8')
            count, _ = self.struct_unpack(file_handle, '@ib')
            _ = self.struct_unpack(file_handle, '@i')
            self.vocab[word] = {
                "index": i,
                "count": count
            }

    def load_vectors(self, file_handle):
        quant = self.struct_unpack(file_handle, '@?')
        (num_vectors, dim) = self.struct_unpack(file_handle, '@2q')
        # Vectors stored by [Matrix::save](https://github.com/facebookresearch/fastText/blob/master/src/matrix.cc)
        assert self.size == dim, 'mismatch between model sizes'
        float_size = struct.calcsize('@f')
        if float_size == 4:
            self.dtype = np.dtype(np.float32)
        elif float_size == 8:
            self.dtype = np.dtype(np.float64)

        self.num_original_vectors = self.bucket
        self.embeddings = np.fromstring(file_handle.read(num_vectors * dim * float_size), dtype=self.dtype)
        self.embeddings = self.embeddings.reshape((num_vectors, dim))[len(self.vocab):,:]
        assert self.embeddings.shape == (self.bucket, self.size), \
            'mismatch between weight matrix shape and vocab/model size'

    def struct_unpack(self, file_handle, fmt):
        num_bytes = struct.calcsize(fmt)
        return struct.unpack(fmt, file_handle.read(num_bytes))

    def compute_ngram_indices(self, word):
        return [self.compute_ngram_index(ngram) for ngram in self.compute_ngrams(word)]

    def compute_ngram_index(self, ngram):
        return self.ft_hash(ngram) % self.bucket

    def transform_input(self, words: list, max_words: int, max_chars=280) -> (np.array, np.array):
        result = np.zeros([max_chars], dtype=self.dtype)
        res_index = 0
        mask = np.zeros([max_words, max_chars], dtype=self.dtype)
        for i, word in enumerate(words):
            if i >= max_words:
                break
            ngrams = self.compute_ngram_indices(word)
            for j, ngram in enumerate(ngrams):
                if res_index >= max_chars:
                    break
                result[res_index] = ngram
                mask[i][res_index] = 1.0 / len(ngrams)
                res_index += 1

        return result, mask

    def compute_ngrams(self, word):
        BOW, EOW = ('<', '>')  # Used by FastText to attach to all words as prefix and suffix
        extended_word = BOW + word + EOW
        ngrams = set()
        for i in range(len(extended_word) - self.min_n + 1):
            for j in range(self.min_n, max(len(extended_word) - self.max_n, self.max_n + 1)):
                ngrams.add(extended_word[i:i + j])
        return ngrams

    @staticmethod
    def ft_hash(string):
        """
        Reproduces [hash method](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc)
        used in fastText.
        """
        # Runtime warnings for integer overflow are raised, this is expected behaviour. These warnings are suppressed.
        old_settings = np.seterr(all='ignore')
        h = np.uint32(2166136261)
        for c in string:
            h = h ^ np.uint32(ord(c))
            h = h * np.uint32(16777619)
        np.seterr(**old_settings)
        return h