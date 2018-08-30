from os import path


class FileProvider:
    def __init__(self, workdir):
        self.model = path.join(workdir, 'model.h5')
        self.model_json = path.join(workdir, 'model.json')
        self.logs = path.join(workdir, 'logs')
        self.input_dir = path.join(workdir, 'input')
        self.semeval_train = path.join(self.input_dir, 'semeval_train')
        self.semeval_test = path.join(self.input_dir, 'semeval_test')
        self.evalita = path.join(self.input_dir, 'evalita_train.json')
        self.evalita_emoji_dist = path.join(self.input_dir, 'emoji_dist.tsv')
        self.evalita_train = path.join(self.input_dir, 'evalita_split_train.json')
        self.evalita_test = path.join(self.input_dir, 'evalita_split_test.json')