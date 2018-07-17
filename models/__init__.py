from models.base_cnn import base_cnn


class ModelDefinition:
    def __init__(self, func, params):
        self.params = params
        self.func = func

    def apply(self, values: dict):
        return self.func(*[values[param] for param in self.params])


def get_model(model: str) -> ModelDefinition:
    models = {
        "base_cnn": ModelDefinition(base_cnn, ["vocabulary_size", "embedding_size", "max_seq_length", "embedding_matrix", "y_dictionary"])
    }

    return models[model]
