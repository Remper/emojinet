from models.base_cnn import base_cnn
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
        "vdcnn": ModelDefinition(vdcnn, ["num_classes", "depth", "sequence_length", "shortcut", "pool_type", "sorted", "use_bias"])
    }

    return models[model]
