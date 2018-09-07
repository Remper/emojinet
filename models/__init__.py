from models.base_lstm_user import base_lstm_user
from models.base_lstm_cnn_user import base_lstm_cnn_user
from models.base_lstm_subword import base_lstm_subword
from models.ensemble_cnn_subword import ensemble_cnn_subword
from models.base_cnn import base_cnn
from models.base_lstm import base_lstm
from models.vdcnn import vdcnn


class ModelDefinition:
    def __init__(self, func, params):
        self.params = params
        self.func = func

    def apply(self, values: dict):
        return self.func(*[values[param] for param in self.params])


def get_model(model: str) -> ModelDefinition:
    models = {
        "base_cnn": ModelDefinition(base_cnn, ["vocabulary_size", "embedding_size", "max_seq_length", "embedding_matrix", "y_dictionary"]),
        "base_lstm": ModelDefinition(base_lstm, ["vocabulary_size", "embedding_size", "max_seq_length", "embedding_matrix", "y_dictionary"]),
        "base_lstm_user": ModelDefinition(base_lstm_user, ["vocabulary_size", "embedding_size", "history_size", "max_seq_length", "embedding_matrix", "y_dictionary"]),
        "base_lstm_cnn_user": ModelDefinition(base_lstm_user, ["vocabulary_size", "embedding_size", "history_size", "max_seq_length", "embedding_matrix", "y_dictionary"]),
        "base_lstm_subword": ModelDefinition(base_lstm_subword, ["vocabulary_size", "embedding_size", "max_char_length", "max_seq_length", "embedding_matrix", "y_dictionary"]),
        "ensemble_cnn_subword": ModelDefinition(ensemble_cnn_subword, ["vocabulary_size", "embedding_size", "max_char_length", "max_seq_length", "embedding_matrix", "y_dictionary"]),
        "vdcnn": ModelDefinition(vdcnn, ["num_classes", "depth", "sequence_length", "shortcut", "pool_type", "sorted", "use_bias"])
    }

    return models[model]
