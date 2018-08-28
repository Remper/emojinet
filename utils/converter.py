import numpy as np


class Converter:
    def __init__(self, sequence_max_length=1024):
        self.alphabet = 'aàbcdeèéfghiìjklmnoòpqrstuùvwxyz0123456789-,;.!?:’\'"/|_#@$%ˆ&*˜‘+=<>()[]{}'
        self.char_dict = {}
        self.sequence_max_length = sequence_max_length
        for i, c in enumerate(self.alphabet):
            self.char_dict[c] = i + 1

    def char2vec(self, text):
        data = np.zeros(self.sequence_max_length)
        for i in range(0, len(text)):
            if i > self.sequence_max_length:
                return data
            elif text[i] in self.char_dict:
                data[i] = self.char_dict[text[i]]
            else:
                # unknown character set to be 0
                data[i] = 0
        return data

    def texts_to_sequences(self, dataset):
        D = np.zeros((len(dataset), self.sequence_max_length))
        for dataset_index, dataset_text in enumerate(dataset):
            D[dataset_index] = self.char2vec(dataset_text.lower())
        return D