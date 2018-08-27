def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    comp_sum = shapes_mem_count + trainable_count + non_trainable_count
    total_memory = 4.0*(batch_size*(shapes_mem_count + non_trainable_count) + trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes, float(shapes_mem_count) / comp_sum, float(trainable_count) / comp_sum,  float(non_trainable_count) / comp_sum