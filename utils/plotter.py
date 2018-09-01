import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class Plotter:
    def __init__(self, model, X_test, Y_test, save_path):

        self.classes = ["❤️", "😂", "😍", "😉", "😊", "😁", "😀", "😘", "😎", "👍", "🤣", "🤔",
                        "💙", "😜", "😱", "💪", "😋", "😅", "😭", "🔝", "💕", "☀️", "💋", "✨", "🌹"]
        #self.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
        #                "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24"]
        self.model = model
        self.X_test = X_test
        self.Y_test = Y_test
        self.save_path = save_path

    def get_predictions(self) -> list:
        return [np.argmax(prediction) for prediction in self.model.predict(self.X_test)]

    def compute_and_save_confusion_matrix(self, normalize=False, title="Confusion matrix"):
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(self.Y_test, self.get_predictions())
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure(figsize=(15, 15), dpi=200)
        if normalize:
            cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes)
        plt.yticks(tick_marks, self.classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cnf_matrix.max() / 2.
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, format(cnf_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cnf_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(self.save_path + "/confusion_matrix.png")


